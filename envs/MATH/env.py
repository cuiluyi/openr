from typing import Optional, Callable
from math_verify import parse, verify

from envs.base_env import CoTEnv
from utils import extract_answer


class Env(CoTEnv):
    sep = "\n\n"

    def __init__(
        self,
        config: dict,
        math_problems: dict,
        llm_gen_fn: Callable,
        reset=True,
        reward_model_fn: Optional[Callable] = None,
    ):
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            reset,
            reward_model_fn,
        )

    def post_process_act(self, action: str):
        if not action.endswith(self.sep):
            action = action.strip() + self.sep

        return action

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        return verify(self.math_problem["answer"], extracted_answer)

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0
