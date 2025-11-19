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

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model (same pattern as extractor.py - works without accelerate)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['name'],
            torch_dtype=torch.float32,  # Always use float32 for numerical stability
            trust_remote_code=True
        )

        # Move to device
        self.model.to(device)
        self.model.eval()

        print(f"Model loaded successfully")

#     def create_prompt(self, question):
#         """Create prompt for math problem solving"""
#         prompt = f"""Solve the following question and give your answer please. Show your work and put your final numerical answer in \\boxed{{}}.

# Problem: {question}

# Solution:"""
#         return prompt

    def create_prompt(self, question):
        """Create prompt for math problem solving"""
        prompt = f"""Solve this math problem. If it can be solved with the given information, show your work and put your final numerical answer in \\boxed{{}}. If it cannot be solved due to insufficient information, state that it cannot be solved.

    Problem: {question}

    Solution:"""
        return prompt

    def create_assess_prompt(self, question):
        """Create prompt for binary assessment"""
        prompt = f"""Can this math problem be solved with the given information? Answer Yes or No and put your answer in \\boxed{{}}.

Problem: {question}

Answer:"""
        return prompt

    def _generate_response(self, prompt):
        """Generate response and count output tokens"""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]

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

        # Count only output tokens
        output_tokens = outputs.shape[1] - input_length

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt
        if prompt in response:
            response = response.replace(prompt, '').strip()

        return response, output_tokens

    def solve(self, question):
        """Solve a math problem"""
        prompt = self.create_prompt(question)
        return self._generate_response(prompt)

    def assess(self, question):
        """Assess if problem can be solved"""
        prompt = self.create_assess_prompt(question)
        return self._generate_response(prompt)

    def batch_solve(self, questions, show_progress=True):
        """Solve multiple problems"""
        responses = []
        iterator = tqdm(questions, desc="Solving problems") if show_progress else questions

        for question in iterator:
            response, tokens = self.solve(question)
            responses.append((response, tokens))

        return responses

    def batch_assess(self, questions, show_progress=True):
        """Assess multiple problems"""
        responses = []
        iterator = tqdm(questions, desc="Assessing problems") if show_progress else questions

        for question in iterator:
            response, tokens = self.assess(question)
            responses.append((response, tokens))

        return responses


if __name__ == '__main__':
    # Test the solver
    print("Testing MathSolver...")

    solver = MathSolver('qwen2.5-math-1.5b', device='cpu', max_new_tokens=100)

    test_question = "If John has 5 apples and gives 2 to Mary, how many does he have left?"

    print(f"\nQuestion: {test_question}")
    
    response, tokens = solver.solve(test_question)
    print(f"Response: {response}")
    print(f"Output tokens: {tokens}")