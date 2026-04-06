from lm_eval.tasks.math500_reasoning.task import Math500Reasoning

class Math50Reasoning(Math500Reasoning):
    def test_docs(self):
        # Override to slice only the first 50 docs for debugging
        return super().test_docs().select(range(50))
