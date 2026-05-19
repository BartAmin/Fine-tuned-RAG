import re
import json
import time
from anthropic import Anthropic, RateLimitError

from config import ANTHROPIC_API_KEY, LLM_MODEL, MAX_TOKENS, MAX_TOKENS_JUDGE, RETRY_WAIT, MAX_RETRIES

client = Anthropic(api_key=ANTHROPIC_API_KEY)


# ── Synthetic question generation ─────────────────────────────────────────────
def generate_questions(user_content):
    system_prompt = """
        You are an expert legal question writer generating questions for legal education and comprehension testing.
        - Read the text carefully.
        - Generate realistic, self-contained questions whose answers can be inferred from the material.
        - Focus on legal rules, exceptions, procedures, and practical application.
        - Turn abstract principles into concrete, realistic scenarios.
        Strict rules:
        - ONLY generate a question if it captures something important that would likely be missed without it.
        - Where multiple questions cover the same legal point, vary their form: use different question types (yes/no, causal, conditional, consequential), different characters, and different sentence structures.
        - Each question must test a distinct legal idea, exception, or application.
        - Questions must be fully self-contained — never refer to "the text" or "the passage".
        - Maximum 5 questions. Fewer is fine. 0 is fine.
        - No answers, explanations, or commentary.
        Output only this JSON:
        {"questions": [{"question": "..."}]}
        If no questions, return: {"questions": []}
    """
    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model      = LLM_MODEL,
                max_tokens = MAX_TOKENS,
                system     = system_prompt,
                messages   = [{"role": "user", "content": user_content}]
            )
            raw_text   = message.content[0].text
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {"questions": []}

        except RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print(f"Rate limit hit — waiting {RETRY_WAIT}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_WAIT)
            else:
                print("Max retries reached, skipping chunk.")
                return {"questions": []}

        except Exception as e:
            print(f"Error calling Claude: {e}")
            return {"questions": []}


# ── RAG answer generation ──────────────────────────────────────────────────────
def generate_answer(question, context):
    system_prompt = f"""
        You are an expert legal assistant. Answer the question concisely and accurately
        based solely on the provided context.
        Guidelines:
        * Base your answer strictly on the provided context.
        * Limit your answer to 50 words +- 20%.
        * If the context is insufficient, state:
          "The provided context does not contain sufficient information to answer this question."
        * Do not make assumptions or use knowledge outside the provided context.
        Context:
        {context}
    """
    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model      = LLM_MODEL,
                max_tokens = MAX_TOKENS,
                system     = system_prompt,
                messages   = [{"role": "user", "content": question}]
            )
            return message.content[0].text

        except RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print(f"Rate limit hit — waiting {RETRY_WAIT}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_WAIT)
            else:
                return "Error: max retries reached."

        except Exception as e:
            print(f"Error calling Claude: {e}")
            return "Error: generation failed."


# ── LLM Judge ─────────────────────────────────────────────────────────────────
def judge_answers(question, ground_truth, result_baseline, result_neural):
    judge_system_prompt = """
        You are an expert legal AI evaluator. Evaluate two RAG system answers (System A and System B)
        to a legal question based on the provided ground truth answer.
        Score each answer on the following metrics (1-5):
        1. Faithfulness: Is the answer factually consistent with the ground truth?
        2. Answer Relevance: Does the answer directly address the question?
        3. Completeness: Does the answer cover all key aspects of the ground truth?
        4. Conciseness: Is the answer free of irrelevant information?
        5. Semantic Similarity: How semantically close is the answer to the ground truth?
        6. Overall: Overall quality score.
        Output ONLY raw JSON (no markdown, no code blocks, no explanation):
        {
            "system_a": {
                "faithfulness": <1-5>,
                "answer_relevance": <1-5>,
                "completeness": <1-5>,
                "conciseness": <1-5>,
                "semantic_similarity": <1-5>,
                "overall": <1-5>,
                "reasoning": "<brief explanation>"
            },
            "system_b": {
                "faithfulness": <1-5>,
                "answer_relevance": <1-5>,
                "completeness": <1-5>,
                "conciseness": <1-5>,
                "semantic_similarity": <1-5>,
                "overall": <1-5>,
                "reasoning": "<brief explanation>"
            }
        }
    """
    judge_user_prompt = f"""
        Question:       {question}
        Ground Truth:   {ground_truth}
        System A Answer:  {result_baseline}
        System B Answer:    {result_neural}
    """
    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model      = LLM_MODEL,
                max_tokens = MAX_TOKENS_JUDGE,
                system     = judge_system_prompt,
                messages   = [{"role": "user", "content": judge_user_prompt}]
            )
            raw_text   = message.content[0].text
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result['system_a'], result['system_b']
            return None, None

        except RateLimitError:
            if attempt < MAX_RETRIES - 1:
                print(f"Rate limit hit — waiting {RETRY_WAIT}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_WAIT)
            else:
                return None, None

        except Exception as e:
            print(f"Error calling Claude: {e}")
            return None, None