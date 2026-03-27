from typing import Dict, List
import lm_eval.api.task
from . import utils

class Math500Reasoning(lm_eval.api.task.ConfigurableTask):
    VERSION = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def doc_to_text(self, doc: dict) -> str:
        # standard problem rendering (system prompt added in context)
        return f"Problem: {doc['problem']}\nAnswer:"

    def doc_to_target(self, doc: dict) -> str:
        # include full reasoning/solution in demonstrations
        return doc["solution"]

    def fewshot_context(self, doc, num_fewshot, rnd=None, description=None, **kwargs):
        # instructions specifying reasoning + boxed answer
        system_prompt = (
            "Please solve the following math problem. Provide your reasoning step-by-step. "
            "The final answer should be put at the end of the response inside \\boxed{}."
        )
        # Prepend system prompt as task description
        return super().fewshot_context(
            doc, 
            num_fewshot, 
            rnd=rnd, 
            description=system_prompt + "\n\n", 
            **kwargs
        )

    def process_results(self, doc: dict, results: List[str]) -> Dict[str, int]:
        # Extract last \boxed content from model response, normalize, and compare to gold
        prediction = results[0]
        retval = 0
        
        # 1. Find the boxed answer in prediction
        extracted_pred_boxed = utils.last_boxed_only_string(prediction)
        
        # 2. Extract gold from solution
        gold_boxed = utils.last_boxed_only_string(doc["solution"])
        gold = utils.remove_boxed(gold_boxed) if gold_boxed else doc.get("answer", "")
        
        if extracted_pred_boxed:
            extracted_pred = utils.remove_boxed(extracted_pred_boxed)
            # 3. Normalizing and comparing (is_equiv handles latex/string normalization)
            if utils.is_equiv(extracted_pred, gold):
                retval = 1
        
        return {"exact_match": retval}
