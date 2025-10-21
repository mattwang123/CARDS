"""
Model inference for math problem solving
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from .config import get_model_config


class MathSolver:
    """
    Load instruction-tuned LLM and solve math problems
    """

    def __init__(self, model_name, device='cpu', max_new_tokens=512, temperature=0.7):
        """
        Args:
            model_name: Name of model from config
            device: 'cpu' or 'cuda'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Get model config
        self.config = get_model_config(model_name)

        print(f"Loading model: {self.config['name']}")
        print(f"Device: {device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['name'],
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['name'],
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map='auto' if device == 'cuda' else None,
            trust_remote_code=True
        )

        if device == 'cpu':
            self.model = self.model.to('cpu')

        self.model.eval()

        print(f"Model loaded successfully")

    def create_prompt(self, question):
        """
        Create prompt for math problem solving

        Args:
            question: Math problem text

        Returns:
            str: Formatted prompt
        """
        prompt = f"""Solve the following question and give your answer please. Show your work and put your final numerical answer in \\boxed{{}}.

Problem: {question}

Solution:"""
        return prompt

    def solve(self, question):
        """
        Solve a single math problem

        Args:
            question: Math problem text

        Returns:
            str: Model response
        """
        prompt = self.create_prompt(question)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the prompt from response (return only generated part)
        if prompt in response:
            response = response.replace(prompt, '').strip()

        return response

    def batch_solve(self, questions, show_progress=True):
        """
        Solve multiple math problems

        Args:
            questions: List of math problem texts
            show_progress: Show progress bar

        Returns:
            list: List of model responses
        """
        responses = []

        iterator = tqdm(questions, desc="Solving problems") if show_progress else questions

        for question in iterator:
            response = self.solve(question)
            responses.append(response)

        return responses


if __name__ == '__main__':
    # Test the solver
    print("Testing MathSolver...")

    # Use small model for testing
    solver = MathSolver('gpt2', device='cpu', max_new_tokens=100)

    test_question = "If John has 5 apples and gives 2 to Mary, how many does he have left?"

    print(f"\nQuestion: {test_question}")
    print("\nGenerating response...")

    response = solver.solve(test_question)

    print(f"\nResponse: {response}")
