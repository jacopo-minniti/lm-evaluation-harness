import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import evaluate as hf_evaluate
import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


import re

def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    predictions = []
    for resp, doc in zip(resps, docs):
        doc_preds = []
        prompt = doc["prompt"]
        
        # Extract the function signature from the prompt (e.g., "def add(")
        # This is a highly reliable way to check if the model repeated the prompt.
        sig_match = re.search(r'(def\s+\w+\s*\()', prompt)
        signature = sig_match.group(1) if sig_match else None

        for r in resp:
            # 1. Extract code. Notice the regex change: [\t ]*\r?\n
            # This consumes ONLY the spaces and the newline immediately after ```python
            # preserving all leading indentation of the actual Python code.
            match = re.search(r'```(?:python)?[\t ]*\r?\n(.*?)```', r, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1)
            else:
                # 2. Try to extract if there's an opening ```python but no closing ```
                match = re.search(r'```(?:python)?[\t ]*\r?\n(.*)', r, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted = match.group(1)
                else:
                    # 3. Fallback: assume the model just generated pure code but might have trailing markdown
                    extracted = r if r.find("```") == -1 else r[:r.find("```")]
            
            # Check if the extracted code includes the function signature
            if signature and signature in extracted:
                final_code = extracted
            else:
                final_code = prompt + extracted

            doc_preds.append(final_code)
        predictions.append(doc_preds)
    return predictions

def take_first_15(dataset):
    return dataset.select(range(15))
