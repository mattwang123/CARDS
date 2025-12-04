"""
Baseline methods for comparison
"""
from models.inference import MathSolver


class BaselineAgent:
    """
    Baseline agents that don't use probe
    """

    def __init__(self, solver, system_prompt_template=None):
        """
        Args:
            solver: MathSolver instance
            system_prompt_template: Optional custom system prompt template
        """
        self.solver = solver

        if system_prompt_template is None:
            self.system_prompt = (
                "This query might be incomplete. If it is insufficient, could you follow up with a concise question "
                "to the user about what might be missing? e.g. 'What is the total number of items or total price?'"
            )
        else:
            self.system_prompt = system_prompt_template

    def process(self, question, baseline_type='just_answer'):
        """
        Process a question using baseline method

        Args:
            question: Question text
            baseline_type: 'just_answer' or 'always_prompt'

        Returns:
            dict: {
                'response': str,
                'tokens_used': int,
                'prompt_appended': bool
            }
        """
        import time
        
        baseline_start = time.time()
        prompt_appended = False

        if baseline_type == 'always_prompt':
            question = f"{question}\n\n{self.system_prompt}"
            prompt_appended = True
        # else: 'just_answer' - use question as-is

        # Solve
        response = self.solver.solve(question)

        # Count tokens
        # NOTE: For cost metrics we only count **agent output tokens**, not
        # prompt/user/system tokens.
        total_tokens = len(self.solver.tokenizer.encode(response))

        baseline_time = time.time() - baseline_start
        print(f"    [TIMING] Baseline ({baseline_type}): {baseline_time:.3f}s")

        return {
            'response': response,
            'tokens_used': total_tokens,
            'prompt_appended': prompt_appended
        }




