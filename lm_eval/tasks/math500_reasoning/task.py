import logging
from typing import Dict, List
import lm_eval.api.task
from . import utils

class Math500Reasoning(lm_eval.api.task.ConfigurableTask):
    VERSION = 2

    def __init__(self, config: dict = None, **kwargs):
        if config:
            config.pop("class", None)
        super().__init__(config=config, **kwargs)

    def doc_to_text(self, doc: dict, doc_to_text: str = None) -> str:
        # Standard problem rendering. We leave the "Answer:" string to gen_prefix
        # so it's placed correctly in both chat and base models.
        return f"Problem: {doc['problem']}"

    def doc_to_target(self, doc: dict, doc_to_target: str = None) -> str:
        # include full reasoning/solution in demonstrations
        return doc["solution"]

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        # instructions specifying reasoning + boxed answer
        system_prompt = (
            "Please solve the following math problem. Provide your reasoning step-by-step. "
            "The final answer should be put at the end of the response inside \\boxed{}."
        )
        # We must pop both system_instruction and gen_prefix to avoid "multiple values" errors
        # as they are already provided in the evaluation loop's kwargs.
        kwargs.pop("system_instruction", None)
        kwargs.pop("gen_prefix", None)
        
        # 🟢 Dynamic Prefix Strategy: 
        # - For base models (no chat template): use "Answer:" as a clear completion trigger.
        # - For instruct models (chat template): use an empty prefix for a clean multi-turn.
        use_chat = kwargs.get("apply_chat_template", False)
        # Note: We add a leading newline for base models to separate it from the problem.
        gen_prefix = "\nAnswer:" if not use_chat else ""

        # 🟢 Pattern Consistency Enforcement:
        # We must ensure few-shot examples use the same prefix ("\nAnswer:") as the target,
        # otherwise base models fail to follow the reasoning format. 
        # Update the config attribute directly as the harness defaults fewshots to no prefix.
        if hasattr(self, "fewshot_cfg") and self.fewshot_cfg:
            self.fewshot_cfg.gen_prefix = gen_prefix

        return super().fewshot_context(
            doc, 
            num_fewshot, 
            system_instruction=system_prompt, 
            gen_prefix=gen_prefix,
            **kwargs
        )

    def process_results(self, doc: dict, results: List[str]) -> Dict[str, float]:
        # Robust evaluation: wrap in try-except to prevent whole run crash on edge cases
        try:
            # Extract last \boxed content from model response, normalize, and compare to gold
            prediction = results[0]
            retval = 0
            
            # 1. Extract gold from solution (Legacy logic)
            gold_boxed = utils.last_boxed_only_string(doc["solution"])
            gold = utils.remove_boxed(gold_boxed) if gold_boxed else doc.get("answer", "")
            
            # 2. Find the boxed answer in prediction
            extracted_pred_boxed = utils.last_boxed_only_string(prediction)
            
            if extracted_pred_boxed:
                extracted_pred = utils.remove_boxed(extracted_pred_boxed)
                # 3. Normalizing and comparing (is_equiv handles latex/string normalization)
                if utils.is_equiv(extracted_pred, gold):
                    retval = 1
            else:
                # 4. Fallback Strategy: check final 10 chars if boxed is missing/incorrect
                if utils.is_equiv_fallback(prediction, gold):
                    retval = 1
            
            results_dict = {"exact_match": float(retval)}

            # 🟢 Augmented logic with math_verify
            if utils.parse is not None and utils.verify is not None:
                # We use the answer field which is processed by process_docs to be wrapped in $
                gold_parsed = utils.parse(doc["answer"])
                # We parse the raw prediction
                pred_parsed = utils.parse(prediction)
                # verify returns a boolean or list of booleans
                is_correct = utils.verify(gold_parsed, pred_parsed)
                results_dict["math_equal_at_1"] = float(is_correct)
            else:
                # Ensure the key exists even if math_verify is missing
                results_dict["math_equal_at_1"] = 0.0
            
            return results_dict
        except Exception as e:
            logging.warning(
                f"Evaluation failed for sample {doc.get('unique_id', 'unknown')} "
                f"with error: {type(e).__name__}: {e}. Marking as incorrect (0)."
            )
            logging.error(f"Prediction that caused crash: {results[0]}")
            return {"exact_match": 0.0, "math_equal_at_1": 0.0}
