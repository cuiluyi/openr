from typing import Optional
from openai import OpenAI
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def extract_answer(completion: str):
    answer_parsed = parse(
        completion,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    return answer_parsed


def verify_answer(answer: str, gold: str) -> bool:
    try:
        gold_parsed = parse(gold)
        answer_parsed = parse(answer)
        flag = verify(gold_parsed, answer_parsed)
    except Exception as e:
        flag = False
    return flag


TEMPLATE = """You are a mathematical answer validator. You will be provided with a mathematical problem and you need to compare the answer in the reference solution, and the final answer in a model's solution to determine if they are equivalent, even if formatted differently.

PROBLEM:

{problem}

REFERENCE SOLUTION:

{answer}

MODEL'S SOLUTION:

{generation}

Focus ONLY on comparing the final mathematical answer provided by the model while ignoring differences in:

- Formatting (e.g., \\boxed{{}} vs plain text)
- Multiple choice formatting (e.g., "A" vs full solution)
- Order of coordinate pairs or solutions
- Equivalent mathematical expressions or notation variations
- If the model's answer is nonsense, return "Verdict: AMBIGUOUS"

Start with a brief explanation of your comparison (2-3 sentences). Then output your final answer in one of the following formats:

- "Verdict: EQUIVALENT"
- "Verdict: DIFFERENT"
- "Verdict: AMBIGUOUS"
"""

MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"

RIGHT_TAG = "Verdict: EQUIVALENT"


def llm_verify_answer(problem: str, answer: str, gold: str) -> bool:
    prompt = TEMPLATE.format(problem=problem, answer=gold, generation=answer)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return RIGHT_TAG in completion.choices[0].message.content


def parser_llm_verify(problem: str, answer: str, gold: str) -> bool:
    parsed_score = verify_answer(answer, gold)
    if parsed_score:
        return parsed_score
    llm_score = llm_verify_answer(problem, answer, gold)
    return llm_score


if __name__ == "__main__":
    problem = "If $\\frac {1}{x} - \\frac {1}{y} = \\frac {1}{z}$, then $z$ equals:\n$\\textbf{(A)}\\ y - x\\qquad \\textbf{(B)}\\ x - y\\qquad \\textbf{(C)}\\ \\frac {y - x}{xy}\\qquad \\textbf{(D)}\\ \\frac {xy}{y - x}\\qquad \\textbf{(E)}\\ \\frac {xy}{x - y}$"
    solution = "1. Start with the given equation:\n   \\[\n   \\frac{1}{x} - \\frac{1}{y} = \\frac{1}{z}\n   \\]\n\n2. Find a common denominator for the left-hand side:\n   \\[\n   \\frac{y}{xy} - \\frac{x}{xy} = \\frac{1}{z}\n   \\]\n\n3. Combine the fractions:\n   \\[\n   \\frac{y-x}{xy} = \\frac{1}{z}\n   \\]\n\n4. To isolate $z$, take the reciprocal of both sides:\n   \\[\n   z = \\frac{1}{\\frac{y-x}{xy}}\n   \\]\n\n5. Simplify the right-hand side:\n   \\[\n   z = \\frac{xy}{y-x}\n   \\]\n\n6. Thus, the value of $z$ is:\n   \\[\n   \\boxed{\\textbf{(D)}\\ \\frac{xy}{y - x}}\n   \\]"
    completion = "<｜begin▁of▁system1｜>\nGiven the equation \\(\\frac{1}{x} - \\frac{1}{y} = \\frac{1}{z}\\), we need to solve for \\(z\\).\n<｜end▁of▁system1｜><｜begin▁of▁system1｜>\nFirst, we find a common denominator for the left side of the equation: \\[\\frac{1}{x} - \\frac{1}{y} = \\frac{y - x}{xy}\\]\n<｜end▁of▁system1｜><｜begin▁of▁system1｜>\nThis simplifies the equation to: \\[\\frac{y - x}{xy} = \\frac{1}{z}\\]\n<｜end▁of▁system1｜><｜begin▁of▁system1｜>\nNext, we cross-multiply to solve for \\(z\\): \\[(y - x)z = xy\\]\n<｜end▁of▁system1｜><｜begin▁of▁system1｜>\nThen, we isolate \\(z\\) by dividing both sides by \\(y - x\\): \\[z = \\frac{xy}{y - x}\\]\n<｜end▁of▁system1｜><｜begin▁of▁system1｜>\nThus, the correct answer is \\(\\boxed{D}\\).\n<｜end▁of▁system1｜><｜end▁of▁sentence｜>"
    print(extract_answer(completion))
    print(verify_answer(completion, solution))
    print(llm_verify_answer(problem, completion, solution))
