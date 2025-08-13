import time

import asyncio
from functools import wraps
from typing import Callable, TypeVar, ParamSpec
from openai import AsyncOpenAI
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from constants import BASE_URL, API_KEY, TEMPLATE, MODEL_NAME, RIGHT_TAG

P = ParamSpec("P")
R = TypeVar("R")


def timing_decorator(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"[TIMER] Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds")
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[TIMER] Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds")
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


@timing_decorator
def verify_answer(answer: str, gold: str) -> bool:
    try:
        gold_parsed = parse(gold, timeout_seconds=None)
        answer_parsed = parse(answer, timeout_seconds=None)
        flag = verify(gold_parsed, answer_parsed, timeout_seconds=None)
    except Exception as e:
        flag = False
    return flag


# global async client
aclient = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# Semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(10)  # Adjust the number as needed


@timing_decorator
async def allm_verify_answer(problem: str, answer: str, gold: str) -> bool:
    prompt = TEMPLATE.format(problem=problem, answer=gold, generation=answer)

    async with semaphore:
        max_retries = 10
        delay_seconds = 3

        for attempt in range(max_retries):
            try:
                completion = await aclient.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=60.0,
                )
                return RIGHT_TAG in completion.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay_seconds)
                    print(f"Retrying due to error: {e}. Attempt {attempt + 1}/{max_retries}")
                else:
                    print(f"Max retries exceeded: {e}")
                    return False


async def async_parser_llm_verify(problem: str, answer: str, gold: str) -> bool:
    parsed_score = verify_answer(answer, gold)
    if parsed_score:
        return parsed_score
    return await allm_verify_answer(problem, answer, gold)
