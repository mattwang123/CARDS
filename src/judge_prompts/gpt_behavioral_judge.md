You are evaluating math problem solving behavior and correctness.

BEHAVIOR CLASSIFICATION:
- "attempted": Model provides mathematical work, calculations, or numerical answer (even if wrong)
- "refused": Model explicitly states it cannot solve, needs more information, or refuses to answer

CORRECTNESS EVALUATION (for attempts only):
- "correct": Final numerical answer matches ground truth
- "incorrect": Final numerical answer is wrong or unclear
- "n/a": For refusals (no answer to evaluate)

Return JSON with:
- "behavior": "attempted" or "refused"
- "correctness": "correct", "incorrect", or "n/a"
- "extracted_answer": numerical answer found or "refused"
- "explanation": brief reasoning