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

        return super().fewshot_context(
            doc, 
            num_fewshot, 
            system_instruction=system_prompt, 
            gen_prefix=gen_prefix,
            **kwargs
        )

    def process_results(self, doc: dict, results: List[str]) -> Dict[str, int]:
        # Extract last \boxed content from model response, normalize, and compare to gold
        prediction = results[0]
        retval = 0
        
        # 1. Extract gold from solution
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
        
        return {"exact_match": retval}
