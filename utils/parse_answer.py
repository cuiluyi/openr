from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from typing import Optional


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


def verify_answer(completion: str, solution: str, **kwargs) -> Optional[float]:
    gold = parse(solution, extraction_mode="first_match")
    
    if len(gold) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer = extract_answer(completion)
        # Compute binary rewards if verifiable, `None` otherwise to skip this example
        try:
            reward = float(verify(gold, answer))
        except Exception as e:
            print(f"verify failed: {e}, answer: {answer}, gold: {gold}")
            reward = float(0)
    else:
        # If the gold solution is not parseable, we assign `None` to skip this example
        reward = float(0)

    return reward
