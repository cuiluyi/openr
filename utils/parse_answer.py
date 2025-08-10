import time
from openai import OpenAI
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from constants import BASE_URL, API_KEY, TEMPLATE, MODEL_NAME, RIGHT_TAG


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
        gold_parsed = parse(gold, timeout_seconds=None)
        answer_parsed = parse(answer, timeout_seconds=None)
        flag = verify(gold_parsed, answer_parsed, timeout_seconds=None)
    except Exception as e:
        flag = False
    return flag


def llm_verify_answer(problem: str, answer: str, gold: str) -> bool:
    prompt = TEMPLATE.format(problem=problem, answer=gold, generation=answer)
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    max_retries = 10
    delay_seconds = 3

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )
            return RIGHT_TAG in completion.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay_seconds)
            else:
                raise e  # raise the last error if all retries failed


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
