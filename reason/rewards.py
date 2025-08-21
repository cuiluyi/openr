"""Reward functions for GRPO training."""

import re
from typing import Sequence

import numpy as np
from tdigest import TDigest

from utils import parser_llm_verify, get_tokens_num


# Precompiled patterns anchored at the current start (\A)
_BLOCK2_RE = re.compile(r"\A<｜begin▁of▁system2｜>\n(.*?)\n<｜end▁of▁system2｜>", re.DOTALL)
_BLOCK1_RE = re.compile(r"\A<｜begin▁of▁system1｜>\n(.*?)\n<｜end▁of▁system1｜>", re.DOTALL)


def _valid_system2_body(body: str) -> bool:
    """system2 body must start with exactly one <think>...</think>, then some content."""
    if not body.startswith("<think>"):
        return 0.0
    m = re.search(r"</think>", body, flags=re.DOTALL)
    if not m:
        return 0.0
    end_idx = m.end()
    # ensure only one pair of think tags
    if "<think>" in body[end_idx:] or "</think>" in body[end_idx:]:
        return 0.0
    # require at least one character after </think> before the final newline/end
    # (if you want to allow empty tail, change to: return end_idx <= len(body))
    return float(end_idx < len(body))


def format_reward(text: str) -> float:
    """
    Return True iff `text` is entirely a concatenation of blocks:
      (1) <｜begin▁of▁system2｜>\n<think>...</think>\n....\n<｜end▁of▁system2｜>
      (2) <｜begin▁of▁system1｜>\n...\n<｜end▁of▁system1｜>
    The order and count are arbitrary.
    """
    s = text
    while s:
        m2 = _BLOCK2_RE.match(s)
        if m2:
            if not _valid_system2_body(m2.group(1)):
                return 0.0
            s = s[m2.end() :]
            continue

        m1 = _BLOCK1_RE.match(s)
        if m1:
            s = s[m1.end() :]
            continue

        # Neither block matches at the current start -> invalid
        return 0.0

    return 1.0


def count_blocks(text: str) -> int:
    """
    Return the number of blocks (type1 or type2) if valid.
    If text is invalid, return -1.
    """
    s = text
    count = 0
    while s:
        m2 = _BLOCK2_RE.match(s)
        if m2:
            if not _valid_system2_body(m2.group(1)):
                return -1
            count += 1
            s = s[m2.end() :]
            continue

        m1 = _BLOCK1_RE.match(s)
        if m1:
            count += 1
            s = s[m1.end() :]
            continue

        # invalid chunk
        return -1

    return count


def reasoning_steps_reward(text: str) -> float:
    count = count_blocks(text)
    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return min(1.0, count / 3)


class LengthReward:
    def __init__(self, update_interval: int = 100):
        """
        Args:
            compression: t-digest compression parameter, affects accuracy and memory usage.
            update_interval: how many calls between quantile updates.
        """
        self.tdigest = TDigest()
        self.min_len = 1000
        self.max_len = 8000
        self.update_interval = update_interval
        self.call_count = 0
        self.q1 = None
        self.median = None
        self.q3 = None

    def _piecewise_reward(self, length: int, q1: float, median: float, q3: float) -> float:
        """
        Piecewise linear reward function:
        - [min_len, q1]    : from  1.0 ->  0.5
        - [q1, median]     : from  0.5 ->  0.0
        - [median, q3]     : from  0.0 -> -0.5
        - [q3, max_len]    : from -0.5 -> -1.0
        """
        assert q1 <= median <= q3

        if length <= q1:
            return 1.0 - 0.5 * (length - self.min_len) / (q1 - self.min_len + 1e-8)
        elif length <= median:
            return 0.5 - 0.5 * (length - q1) / (median - q1 + 1e-8)
        elif length <= q3:
            return 0.0 - 0.5 * (length - median) / (q3 - median + 1e-8)
        else:
            return -0.5 - 0.5 * (length - q3) / (self.max_len - q3 + 1e-8)

    def __call__(self, text: str, is_correct: bool) -> float:
        length = get_tokens_num(text)
        self.tdigest.update(length)

        self.min_len = min(self.min_len, length)
        self.max_len = max(self.max_len, length)
        self.call_count += 1

        # Update quantiles periodically
        if self.call_count % self.update_interval == 0 and self.call_count >= 10:
            self.q1 = self.tdigest.percentile(25)
            self.median = self.tdigest.percentile(50)
            self.q3 = self.tdigest.percentile(75)

        # If quantiles not ready, fallback to simple linear reward
        if self.q1 is None or self.median is None or self.q3 is None:
            reward = 1.0 - (length / (self.max_len if self.max_len > 0 else 1))
        else:
            reward = self._piecewise_reward(length, self.q1, self.median, self.q3)

        return reward if is_correct else min(0.0, reward)

REWARD_FUNCS_REGISTRY = {
    "accuracy": parser_llm_verify,
    "format": format_reward,
    "reasoning_steps": reasoning_steps_reward,
    "length": LengthReward(),
}

# Define weights for each reward function
REWARD_WEIGHTS = {
    "accuracy": 0.4,
    "format": 0.1,
    "reasoning_steps": 0.1,
    "length": 0.4,
}


def get_final_reward(
    reward_func_names: list[str],
    problem: str,
    answer: str,
    gold: str,
) -> float:
    """
    Compute the final weighted reward as a single float.
    """
    rewards = []
    is_correct = parser_llm_verify(problem, answer, gold)

    for func_name in reward_func_names:
        if func_name == "accuracy":
            reward_val = float(is_correct)
        elif func_name == "length":
            reward_val = REWARD_FUNCS_REGISTRY[func_name](answer, is_correct)
        else:
            reward_val = REWARD_FUNCS_REGISTRY[func_name](answer)
        rewards.append(reward_val)

    # Compute weighted sum
    weighted_reward = sum(
        rewards[i] * REWARD_WEIGHTS[func_name] for i, func_name in enumerate(reward_func_names)
    )

    return float(weighted_reward)
