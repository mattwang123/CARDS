"""
Workflow agent that uses probe to detect insufficient questions
"""
import torch
from models.inference import MathSolver


class WorkflowAgent:
    """
    Agent that uses probe to decide whether to append system prompt
    """

    def __init__(self, probe_inference, solver, system_prompt_template=None):
        """
        Args:
            probe_inference: ProbeInference instance
            solver: MathSolver instance
            system_prompt_template: Optional custom system prompt template
        """
        self.probe_inference = probe_inference
        self.solver = solver

        if system_prompt_template is None:
            self.system_prompt = (
                "This query might be incomplete. Could you follow up with a concise question "
                "to the user about what might be missing? e.g. 'What is the total number of items or total price?'"
            )
        else:
            self.system_prompt = system_prompt_template

    def process(self, question):
        """
        Process a question through the workflow

        Args:
            question: Question text

        Returns:
            dict: {
                'response': str,
                'tokens_used': int,
                'probe_prediction': bool,  # True if sufficient, False if insufficient
                'probe_confidence': float,
                'prompt_appended': bool
            }
        """
        import time
        
        # Run probe prediction
        probe_start = time.time()
        is_sufficient, confidence = self.probe_inference.predict(question)
        probe_time = time.time() - probe_start
        print(f"    [TIMING] Probe prediction: {probe_time:.3f}s")

        # Modify question if insufficient
        prompt_appended = False
        if not is_sufficient:
            question = f"{question}\n\n{self.system_prompt}"
            prompt_appended = True

        # Solve
        solver_start = time.time()
        response = self.solver.solve(question)
        solver_time = time.time() - solver_start
        print(f"    [TIMING] Solver generation: {solver_time:.3f}s")

        # Count tokens
        # NOTE: For cost metrics we only count **agent output tokens**, not
        # prompt/user/system tokens. This keeps the metric focused on model
        # generation cost.
        total_tokens = len(self.solver.tokenizer.encode(response))

        return {
            'response': response,
            'tokens_used': total_tokens,
            'probe_prediction': bool(is_sufficient),  # Convert numpy bool to Python bool
            'probe_confidence': float(confidence),  # Convert numpy float to Python float
            'prompt_appended': bool(prompt_appended)  # Ensure Python bool
        }




