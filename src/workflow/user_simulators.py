"""
User simulators for workflow experiments
"""
import os
import sys
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


class UserSimulator:
    """
    Base class for user simulators
    """

    def respond(self, agent_response, original_question, missing_info):
        """
        Decide whether to disclose missing information

        Args:
            agent_response: Agent's response to the incomplete question
            original_question: Original complete question (before removal)
            missing_info: The missing information (removed_value or removed_description)

        Returns:
            tuple: (response: str or None, disclosed: bool)
                - response: The missing info if disclosed, None otherwise
                - disclosed: Whether the user decided to disclose the information
        """
        raise NotImplementedError


class GPT5Simulator(UserSimulator):
    """
    GPT-5 simulator using OpenAI API (gpt-4o as proxy)
    """

    def __init__(self, model='gpt-4o', temperature=0.7):
        """
        Args:
            model: OpenAI model to use (default: gpt-4o)
            temperature: Sampling temperature
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def respond(self, agent_response, original_question, missing_info):
        """
        Use GPT to simulate user behavior
        """
        import time
        
        prompt_start = time.time()
        prompt = f"""You are a user interacting with a math problem solver. You started with an incomplete question, and the solver has responded.

Your original complete question was: {original_question}

The missing information is: {missing_info}

The solver responded: {agent_response}

Decide whether to disclose the missing information. You should only disclose it if the solver explicitly asks about what's missing. If the solver attempts to answer the problem without asking, you should not disclose the information.

Respond with ONLY the missing information if you decide to disclose it, or respond with "None" if you don't disclose it."""
        prompt_time = time.time() - prompt_start

        api_start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=100
        )
        api_time = time.time() - api_start

        user_response = response.choices[0].message.content.strip()
        
        total_time = time.time() - prompt_start
        print(f"    [TIMING] GPT5Simulator - prompt: {prompt_time:.3f}s, API: {api_time:.3f}s, total: {total_time:.3f}s")
        
        if user_response.lower() == "none" or user_response.lower() == "none.":
            return (None, False)
        return (user_response, True)


class RAGSimulator(UserSimulator):
    """
    RAG simulator: computes cosine similarity between agent's query and missing_info.
    Returns missing_info probabilistically based on similarity.
    """

    def __init__(self, embedding_model='text-embedding-3-small'):
        """
        Args:
            embedding_model: OpenAI embedding model (default: 'text-embedding-3-small')
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model

    def _embed_text(self, text):
        """Extract embedding using OpenAI embeddings API"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)

    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return np.dot(vec1_norm, vec2_norm)

    def respond(self, agent_response, original_question, missing_info):
        """
        Compute cosine similarity between agent's query and missing_info.
        Return missing_info probabilistically based on similarity.
        
        Args:
            agent_response: Agent's response (the follow-up question)
            original_question: Original complete question (not used)
            missing_info: The missing information for this specific example
        
        Returns:
            str or None: Missing info if similarity-based probability succeeds, None otherwise
        """
        # Embed agent's query and missing_info
        query_embedding = self._embed_text(agent_response)
        missing_embedding = self._embed_text(missing_info)
        
        # Compute cosine similarity
        similarity = self._cosine_similarity(query_embedding, missing_embedding)
        
        # Success rate proportional to similarity
        # Map similarity from [-1, 1] to [0, 1] for probability
        normalized_similarity = (similarity + 1) / 2
        retrieval_prob = max(0.0, min(1.0, normalized_similarity))
        
        # Decide whether to retrieve (probabilistic based on similarity)
        import random
        should_retrieve = random.random() < retrieval_prob
        
        if should_retrieve:
            return (missing_info, True)
        return (None, False)


class GPT5RecallSimulator(UserSimulator):
    """
    GPT-5 with recall tool: GPT generates a query, then RAG retrieves based on OpenAI embeddings cosine similarity
    """

    def __init__(self, model='gpt-4o', temperature=0.7, embedding_model='text-embedding-3-small'):
        """
        Args:
            model: OpenAI model for GPT-5 simulation
            temperature: Sampling temperature
            embedding_model: OpenAI embedding model (default: 'text-embedding-3-small')
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.embedding_model = embedding_model

    def _embed_text(self, text):
        """Extract embedding using OpenAI embeddings API"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)

    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return np.dot(vec1_norm, vec2_norm)

    def _rag_retrieve(self, query, missing_info):
        """
        RAG retrieval: compute similarity between query and missing_info using OpenAI embeddings
        
        Returns:
            tuple: (retrieved_info or None, similarity_score)
        """
        import time
        
        # Embed query and missing_info
        embed_start = time.time()
        query_embedding = self._embed_text(query)
        missing_embedding = self._embed_text(missing_info)
        embed_time = time.time() - embed_start
        
        # Compute similarity
        similarity = self._cosine_similarity(query_embedding, missing_embedding)
        
        # Success rate proportional to similarity (probabilistic)
        # Normalize similarity to [0, 1] range (OpenAI embeddings typically give similarity in [-1, 1])
        # For better retrieval, we can use (similarity + 1) / 2 or just use similarity directly
        # Since cosine similarity is already in [-1, 1], we'll map it to [0, 1] for probability
        normalized_similarity = (similarity + 1) / 2  # Map from [-1, 1] to [0, 1]
        retrieval_prob = max(0.0, min(1.0, normalized_similarity))
        
        import random
        should_retrieve = random.random() < retrieval_prob
        
        return (missing_info, similarity) if should_retrieve else (None, similarity)

    def respond(self, agent_response, original_question, missing_info):
        """
        Use GPT-5 to generate a query, then use RAG to retrieve based on cosine similarity
        """
        import time
        
        total_start = time.time()
        
        # Step 1: GPT generates a query for RAG retrieval
        query_prompt_start = time.time()
        query_prompt = f"""You are a user with access to a RAG (Retrieval-Augmented Generation) system. 
The solver has responded to your incomplete question: {agent_response}

You need to retrieve the missing information: {missing_info}

Generate a concise query (1-5 words) that would help retrieve this information from the RAG system.
Respond with ONLY the query, nothing else."""
        query_prompt_time = time.time() - query_prompt_start

        query_api_start = time.time()
        query_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query_prompt}],
            temperature=self.temperature,
            max_tokens=20
        )
        query_api_time = time.time() - query_api_start
        
        gpt_query = query_response.choices[0].message.content.strip()
        
        # Step 2: Use RAG to retrieve based on cosine similarity
        rag_start = time.time()
        retrieved_info, similarity = self._rag_retrieve(gpt_query, missing_info)
        rag_time = time.time() - rag_start
        
        total_time = time.time() - total_start
        query_gen_time = query_prompt_time + query_api_time
        print(f"    [TIMING] GPT5RecallSimulator - query_gen: {query_gen_time:.3f}s, RAG_retrieve: {rag_time:.3f}s, similarity: {similarity:.3f}, total: {total_time:.3f}s")
        print(f"      GPT query: '{gpt_query}'")
        
        disclosed = retrieved_info is not None
        return (retrieved_info, disclosed)

