vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --trust-remote-code


curl http://e03:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful research assistant."},
      {"role": "user", "content": "Briefly explain the benefit of PagedAttention for LLM inference."}
    ],
    "temperature": 0.7
  }'