import re


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    # Metric computation is skipped; eval is done externally via scripts/eval_humaneval.py
    return 0.0


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    predictions = []
    for resp, doc in zip(resps, docs):
        doc_preds = []
        prompt = doc["prompt"]

        sig_match = re.search(r'(def\s+\w+\s*\()', prompt)
        signature = sig_match.group(1) if sig_match else None

        for r in resp:
            match = re.search(r'```(?:python)?[\t ]*\r?\n(.*?)```', r, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1)
            else:
                match = re.search(r'```(?:python)?[\t ]*\r?\n(.*)', r, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted = match.group(1)
                else:
                    extracted = r if r.find("```") == -1 else r[:r.find("```")]

            if signature and signature in extracted:
                final_code = extracted
            else:
                final_code = prompt + extracted

            doc_preds.append(final_code)
        predictions.append(doc_preds)
    return predictions


def take_first_15(dataset):
    return dataset.select(range(15))
