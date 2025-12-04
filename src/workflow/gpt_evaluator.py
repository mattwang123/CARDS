"""
GPT-based evaluator for workflow experiments using GPT-4o-mini
"""
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv


class GPTEvaluator:
    """
    GPT-4o-mini–based evaluator for workflow behavior.

    This is the **only** evaluator used by the workflow. It is responsible for:
      - Judging **behavior_correctness**: did the agent ask vs answer appropriately,
        given whether the question is sufficient or insufficient? (1 = correct, 0 = incorrect)
      - Judging **answer correctness** when a ground-truth numeric answer is
        provided.
      - For insufficient questions, judging whether the clarification question
        the agent asks actually targets the true missing information.

    The evaluator always operates on the **first agent response** (the initial
    message produced by the workflow agent or baseline).
    """

    def __init__(self, openai_client=None):
        """
        Args:
            openai_client: OpenAI client instance (if None, will create from env)
        """
        if openai_client is None:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment. Set it in .env file.")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = openai_client

        # Load judge prompt
        prompt_path = Path(__file__).parent.parent / "judge_prompts" / "gpt_workflow_judge.md"
        with open(prompt_path, 'r') as f:
            self.judge_prompt = f.read()

    def evaluate(
        self,
        question,
        agent_response,
        ground_truth_sufficient,
        missing_info=None,
        ground_truth_answer=None,
    ):
        """
        Evaluate a single agent response using GPT-4o-mini.

        Args:
            question: Original question text.
            agent_response: Agent's **first** response text.
            ground_truth_sufficient: True if the original question is actually
                sufficient according to the dataset.
            missing_info: Optional string describing the true missing
                information for insufficient questions (e.g., `removed_value`
                or `removed_description`).
            ground_truth_answer: Optional ground-truth numeric answer string
                (e.g. "#### 27") when available.

        Returns:
            dict with at least:
                - 'behavior_correctness': float in {0.0, 1.0}  # 1 = correct behavior, 0 = incorrect
                - 'answer_attempted': bool          # did the agent clearly try to give a numeric answer?
                - 'answer_correct': bool or None    # ONLY meaningful when ground_truth_sufficient is True
                - 'asked_question': bool            # did the agent explicitly ask the user anything?
                - 'question_matches_missing_info': bool or None
                    # for insufficient questions: does the clarification question target the true missing info?
                    # for sufficient questions: always None
                - 'gpt_judgment': dict  # Full raw JSON judgment from GPT
        """
        # For **sufficient** questions, we expect to always have a ground-truth
        # numeric answer for correctness checking. If it is missing, this is a
        # configuration / data error and we fail fast.
        if ground_truth_sufficient and ground_truth_answer is None:
            raise ValueError(
                "GPTEvaluator.evaluate called with ground_truth_sufficient=True "
                "but ground_truth_answer=None. For sufficient questions you "
                "must provide a ground-truth answer (e.g., GSM8K 'answer')."
            )

        # Build different instructions for sufficient vs insufficient ground truth.
        if ground_truth_sufficient:
            # Sufficient-question evaluator: judge behavior_correctness + answer correctness.
            gt_answer_text = ground_truth_answer or "UNKNOWN"
            evaluation_prompt = f"""{self.judge_prompt}

## YOUR TASK

You are evaluating the behavior of an agent answering a **sufficient** math word problem.

**Question**:
{question}

**Ground-Truth Final Answer (if provided)**:
{gt_answer_text}

**Agent Response**:
{agent_response}

You must infer the following fields:

1. "behavior_correctness" (0 or 1) – behavior quality for a **SUFFICIENT** question:
   - behavior_correctness = 1 (correct) if the agent directly attempts to solve / answer the
     problem. It may restate or paraphrase the question, but it should NOT ask
     for extra information that is not actually needed.
   - behavior_correctness = 0 (incorrect) if the agent asks the user for more information or
     otherwise treats the question as incomplete, even though it is already
     sufficient.

2. "answer_attempted" (boolean):
   - true  → the agent clearly presents or implies a specific numeric answer
             (e.g., "the answer is 12", "so we get 7", "there are 15 in total").
   - false → the agent does not commit to any numeric answer.

3. "answer_correct" (true / false / null):
   - true   → the numeric answer in the agent's response exactly matches the
              ground-truth final answer (ignoring formatting like "#### 27"
              vs "27").
   - false  → the agent provides a numeric answer that does NOT equal the
              ground-truth answer.
   - null   → the agent did not provide any usable numeric answer.

4. "asked_question" (boolean):
   - true  → the agent explicitly asks the user something (e.g., contains a
             clear question mark and a request for information).
   - false → no explicit question to the user.

5. "question_matches_missing_info" (always null for sufficient questions).

Return JSON with at least:
- "behavior_correctness": 1 or 0        # 1 = correct behavior, 0 = incorrect behavior
- "answer_attempted": boolean
- "answer_correct": true/false/null
- "asked_question": boolean
- "question_matches_missing_info": null   # always null for sufficient questions

Respond with JSON only."""
        else:
            # Insufficient-question evaluator: judge behavior_correctness + matching missing info.
            # For insufficient questions, we MUST be given the true missing
            # information (e.g., removed_value / removed_description). If it is
            # missing, this is a data/configuration error.
            if missing_info is None:
                raise ValueError(
                    "GPTEvaluator.evaluate called with ground_truth_sufficient=False "
                    "but missing_info=None. For insufficient questions you must "
                    "provide the true missing information (e.g., removed_value or "
                    "removed_description)."
                )

            missing_info_text = missing_info
            evaluation_prompt = f"""{self.judge_prompt}

## YOUR TASK

You are evaluating the behavior of an agent answering an **insufficient** math word problem.

**Question**:
{question}

**True Missing Information (research ground truth)**:
{missing_info_text}


**Agent Response**:
{agent_response}

You must infer the following fields:

1. "behavior_correctness" (0 or 1) – behavior quality for an **INSUFFICIENT** question:
   - behavior_correctness = 1 (correct) if the agent recognizes the problem is incomplete and
     asks the user a clarification question (it should NOT commit to a final
     numeric answer before the missing information is provided).
   - behavior_correctness = 0 (incorrect) if the agent attempts to solve / give a numeric answer
     without asking for the missing information, or otherwise behaves as if the
     question were sufficient.

2. "answer_attempted" (boolean):
   - true  → the agent clearly presents or implies a specific numeric answer.
   - false → the agent does not commit to any numeric answer.


3. "asked_question" (boolean):
   - true  → the agent explicitly asks the user something (a clarification or
             request for missing information).
   - false → no explicit question to the user.

4. "question_matches_missing_info" (true / false / null):
   - true   → the agent's clarification question directly addresses the true
              missing information described above.
   - false  → the question is about something else (wrong or irrelevant).
   - null   → the agent did not ask any clarification question.

Return JSON with at least:
- "behavior_correctness": 1 or 0
- "answer_attempted": boolean
- "asked_question": boolean
- "question_matches_missing_info": true/false/null

Respond with JSON only."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a workflow evaluator. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # Deterministic
                max_tokens=300
            )

            judgment = json.loads(response.choices[0].message.content)
            
            # behavior_correctness must be provided explicitly in the JSON output. If it
            # is missing or not parseable, we fail fast to surface a prompt /
            # schema mismatch instead of silently guessing.
            # Support both old "redundancy" and new "behavior_correctness" for backward compatibility
            if "behavior_correctness" not in judgment and "redundancy" not in judgment:
                raise ValueError(
                    f"GPTEvaluator: 'behavior_correctness' field missing from judgment: {judgment}"
                )
            try:
                behavior_correctness = float(judgment.get("behavior_correctness", judgment.get("redundancy")))
            except Exception as e:
                raise ValueError(
                    f"GPTEvaluator: could not parse 'behavior_correctness' as float "
                    f"from judgment {judgment}: {e}"
                )

            # answer_correct is only meaningful for sufficient questions; for
            # insufficient ones we ignore any value the model might return.
            if ground_truth_sufficient:
                answer_correct = judgment.get('answer_correct', None)
            else:
                answer_correct = None

            return {
                'behavior_correctness': float(behavior_correctness),
                'answer_attempted': judgment.get('answer_attempted', judgment.get('attempted_answer', False)),
                'answer_correct': answer_correct,
                'asked_question': judgment.get('asked_question', False),
                'question_matches_missing_info': judgment.get('question_matches_missing_info', None),
                'gpt_judgment': judgment
            }

        except Exception as e:
            print(f"  ⚠ Error in GPT-4o-mini judgment: {e}")
            return {
                'behavior_correctness': 0.0,
                'answer_attempted': False,
                'answer_correct': None,
                'asked_question': False,
                'question_matches_missing_info': None,
                'gpt_judgment': {
                    'verdict': 'error',
                    'explanation': f"API error: {str(e)}"
                }
            }

    def evaluate_answer_only(self, question, agent_response, ground_truth_answer):
        """
        Evaluate **only** numeric answer correctness for a (now complete) question.

        This mode assumes the question has all necessary information and the
        agent is attempting to provide a final answer. It does NOT judge
        behavior_correctness or sufficiency, only whether the numeric answer (if any) in
        `agent_response` matches `ground_truth_answer`.

        Args:
            question: Original (complete) question text. Used for context only.
            agent_response: Agent's final response text (after any follow-up).
            ground_truth_answer: Ground-truth final answer string (e.g. "#### 27").
                This argument is **required**; if it is missing, the caller
                has misconfigured the evaluation pipeline.

        Returns:
            dict: {
                'answer_attempted': bool or None,
                'answer_correct': bool or None,
                'gpt_judgment': dict
            }
        """
        if ground_truth_answer is None:
            raise ValueError(
                "GPTEvaluator.evaluate_answer_only called with ground_truth_answer=None. "
                "This mode requires a concrete ground-truth answer string."
            )

        evaluation_prompt = f"""{self.judge_prompt}

## YOUR TASK

You are evaluating whether the agent's final numeric answer to a COMPLETE math
word problem is correct.

**Question (for context)**:
{question}

**Ground-Truth Final Answer**:
{ground_truth_answer}

**Agent Final Response**:
{agent_response}

You must infer:
- Whether the agent clearly attempted to provide a numeric answer.
- Whether that numeric answer is **exactly equal** to the ground-truth answer
  above (you may ignore formatting like "#### 27" vs "27").

Return JSON with at least:
- "answer_attempted": boolean
- "answer_correct": true/false/null   # null if the response does not contain a usable numeric answer

Respond with JSON only."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a workflow evaluator. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # as deterministic as possible
                max_tokens=200
            )

            judgment = json.loads(response.choices[0].message.content)

            return {
                'answer_attempted': judgment.get('answer_attempted', None),
                'answer_correct': judgment.get('answer_correct', None),
                'gpt_judgment': judgment
            }

        except Exception as e:
            print(f"  ⚠ Error in GPT-4o-mini answer-only judgment: {e}")
            return {
                'answer_attempted': None,
                'answer_correct': None,
                'gpt_judgment': {
                    'verdict': 'error',
                    'explanation': f"API error in answer-only mode: {str(e)}"
                }
            }