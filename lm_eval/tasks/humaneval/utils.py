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
        for r in resp:
            # 1. Try to extract code between ```python and ```
            match = re.search(r'```(?:python)?\s*(.*?)```', r, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1)
            else:
                # 2. Try to extract if there's an opening ```python but no closing ```
                match = re.search(r'```(?:python)?\s*(.*)', r, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted = match.group(1)
                else:
                    # 3. Fallback: assume the model just generated pure code but might have trailing markdown
                    extracted = r if r.find("```") == -1 else r[:r.find("```")]
            
            # If the model repeated the prompt inside the block, conditionally strip it
            if extracted.lstrip().startswith(prompt.strip()):
                extracted = extracted.lstrip()[len(prompt.strip()):]
                if extracted.startswith('\n'):
                    extracted = extracted[1:]

            doc_preds.append(prompt + extracted)
        predictions.append(doc_preds)
    return predictions

def take_first_15(dataset):
    return dataset.select(range(15))
