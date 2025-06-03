import abc
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
import numpy as np
import copy
import pdb
import torch
from distributed.utils import print_with_rank
from transformers import PreTrainedTokenizer
from reason.inference.lm_call import LMCallingConfig, ConcatedLMGenResult

# from envs.diversity_filter import DiversityFilter
from envs.correct_filter import CorrectFilter

INVALID_ANS = "[invalid]"


class NoLegalActionException(Exception):
    pass


class ResetException(Exception):
    pass


class BaseEnv(abc.ABC):
    """Basic environment to use for MCTS"""

    @abc.abstractmethod
    def reset(self, update_legal_action: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def legal_actions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    @staticmethod
    def build_query_str(
        cot_task_desc: Optional[str],
        cot_examples: Optional[str],
        problem_format_str: str,
        problem_input: str,
        is_few_shot: bool = False,
    ):
        """a wrap function that wrap the problem text with certrain format
        e.g. prompt_str = "Input: " + join_numbers(" ", xs) + "\nSteps:\n"
        >>> query_str = Game24Env.build_query_str("1 1 1 1")
        >>> print(query_str)
        >>> Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
        Input: 1 1 1 1
        Steps:

        >>>
        """

        ret = ""
        if cot_task_desc:
            ret += cot_task_desc + "\n"
        if is_few_shot:
            ret += cot_examples + "\n"
        ret += problem_format_str.format(question=problem_input)

        return ret

    @staticmethod
    def build_response_str(
        answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool
    ):
        raise NotImplementedError


class CoTEnv(BaseEnv):
    """The basic environment for solving natural language problems using CoT"""

    sep: str

    @property
    def stop_str(self):
        return self._stop_str

    def _is_correct(self, completion) -> bool:
        raise NotImplementedError

    def get_reward(self):
        """To implement based on learned reward model"""
        raise NotImplementedError

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        task_desc_str: str,
        cot_example_str: str,
        problem_format_str: str,
        reset=True,
        reward_model_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.mcts_mode = "play_with_bot_mode"
        self.math_problems = math_problems
        self.llm_gen_fn = llm_gen_fn
        self.action_history = None
        self.values = []
        self.math_problem = None
        self._legal_actions = None
        self.is_few_shot = config.get("is_few_shot", False)

        self._task_desc_str = task_desc_str
        self._cot_example_str = cot_example_str
        self._problem_format_str = problem_format_str
        self._stop_str = config.get("stop_str", None)

        prefixes = []
        if self._task_desc_str is not None:
            prefixes.append(self._task_desc_str)
        if self.is_few_shot:
            prefixes.append(self._cot_example_str)
        if len(prefixes) > 0:
            self.task_prefix = "\n".join(prefixes)
        else:
            self.task_prefix = None

        self.reward_model_fn = reward_model_fn

        if reset:
            self.reset(update_legal_action=True)

    def reset(self, update_legal_action=True):
        # reset environment to problem idx
        self.set_problem(idx=0)
        self.action_history = []
        self.values = []
        self._init_query = self.build_query_str(
            cot_examples=self._cot_example_str,
            cot_task_desc=self._task_desc_str,
            problem_format_str=self._problem_format_str,
            problem_input=self.math_problem["question"],
            is_few_shot=self.is_few_shot,
        )
        api_completion_token = 0
        if update_legal_action:
            cnt = 0
            while cnt < 5:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = (
                        self.update_legal_actions()
                    )
                    break
                except NoLegalActionException as e:
                    if cnt == 5:
                        raise ResetException
        info = {"api_completion_token": api_completion_token}
        return self.get_state(), info

    def step(self, action, value=0, update_legal_action=True):
        self.action_history.append(action)
        self.values.append(value)
        state = self.get_state()
        reward = self.get_reward()
        terminated, truncated, info = self.get_done_and_info()
        # update legal actions
        if not (terminated or truncated) and update_legal_action:
            cnt = 0
            while cnt < 5:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = (
                        self.update_legal_actions()
                    )
                    info["api_completion_token"] = api_completion_token
                    break
                except NoLegalActionException as e:
                    if cnt == 5:
                        terminated = True
                        reward = 0
                        self._legal_actions = None
                        info["winner"] = 2
                        info["api_completion_token"] = 0
                    else:
                        pass
        else:
            self._legal_actions = None
            if info["winner"] == 1:
                reward = 1.0
            info["api_completion_token"] = 0
        return state, reward, terminated, truncated, info

    def get_state(self):
        # not join about sep_str here because we let vllm return with sep_str
        ret = self._init_query + "".join(
            item for item in self.action_history if item is not None
        )
        return ret

    def post_process_act(self, action: str):
        # This step may change the token count
        return action

    def update_legal_actions(self):
        # # diversity_filter = DiversityFilter()
        # correct_filter = CorrectFilter(self, self.reward_model_fn)

        # try_num = 10
        # for i in range(try_num):
        #     result: ConcatedLMGenResult = self.llm_gen_fn(
        #         input_str=self.get_state(),
        #         config=LMCallingConfig(
        #             n=self.config["max_actions"],
        #             stop_str=self.sep,
        #             include_stop_str_in_output=True,
        #             **self.config["generation_config"]
        #         ),
        #     )
        #     if isinstance(result.finish_reason, str):
        #         result.finish_reason = [result.finish_reason]

        #     # result = diversity_filter.filter(result)
        #     result = correct_filter.filter(result)
        #     if result is not None:
        #         break
        #     if i == try_num - 1 and result is None:
        #         result = correct_filter.max_step
        #         # if correct_filter.max_prm_value < 0.8:
        #         #     raise NoLegalActionException("No possible action have been generated.")

        result: ConcatedLMGenResult = self.llm_gen_fn(
            input_str=self.get_state(),
            config=LMCallingConfig(
                n=self.config["max_actions"],
                stop_str=self.sep,
                include_stop_str_in_output=True,
                **self.config["generation_config"]
            ),
        )
        if isinstance(result.finish_reason, str):
            result.finish_reason = [result.finish_reason]

        texts = result.text
        logps_avg_by_len = result.logp_avg_by_len
        token_len = result.num_tokens
        text_list, prob_list, num_token_list = [], [], []
        finish_reason_list = []
        next_state_terminated = {}

        for i in range(len(texts)):
            # XXX: this process can be improve or moved to other place
            # this is a pre-judge of terminal flag or certain action, by
            # whether the text-generation is stop by the <eos> or stop_str

            terminated = not texts[i].endswith(self.sep)

            processed_act = self.post_process_act(texts[i])
            if (
                len(processed_act) > 0
                and processed_act not in text_list
                # only stop is valid, otherwise the output action is truncated actually
                and result.finish_reason[i] == "stop"
            ):
                text_list.append(processed_act)
                prob_list.append(logps_avg_by_len[i])
                num_token_list.append(token_len[i])
                finish_reason_list.append(result.finish_reason[i])
                next_state_terminated[processed_act] = terminated

        if len(prob_list) == 0:
            print_with_rank("state: {}".format(self.get_state()))
            print_with_rank("gen_result: {}".format(result))
            raise NoLegalActionException("No possible action have been generated.")

        # from .utils import entropy_from_logprobs
        # relative_entropy = entropy_from_logprobs(prob_list)
        
        prob_list = np.exp(prob_list)
        prob_list = np.array(prob_list)
        # normalize probability
        prob_list = prob_list / np.sum(prob_list)
        

        _legal_actions = [
            {
                "action": action,
                "prob": prob,
                "num_token": n_token,
                "finish_reason": finish_reason,
                # "weight": relative_entropy,
            }
            for action, prob, n_token, finish_reason in zip(
                text_list,
                prob_list,
                num_token_list,
                finish_reason_list,
            )
        ]
        self._next_state_terminated = next_state_terminated
        return _legal_actions, result.completion_tokens

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    @property
    def query(self):
        return self._init_query

    @property
    def question(self) -> str:
        return self.math_problem["question"]

    @property
    def answer(self):
        return "".join(self.action_history)

    def get_done_and_info(self):
        info = {"winner": 0}
        try:
            # done when reaches maximum length or LLM generates stop words
            if self.stop_str is not None and self.stop_str in self.action_history[-1]:
                terminated = True
            elif self._next_state_terminated[self.action_history[-1]]:
                terminated = True
            elif self.sep not in self.action_history[-1]:
                # This is because the output is stopped by eos
                terminated = True
            else:
                terminated = False
        except Exception as e:
            print(self)

        truncated = len(self.action_history) >= self.config["max_length"]
        assert len(self.action_history) <= self.config["max_length"]
        if terminated or truncated:
            # if self._is_correct(self.action_history[-1]):
            #     info["winner"] = 1
            # else:
            #     info["winner"] = 2
            info["winner"] = 0
            return terminated, truncated, info
        return terminated, truncated, info

    def copy(self):
        env = self.__class__(
            config=self.config,
            math_problems=self.math_problems,
            llm_gen_fn=self.llm_gen_fn,
            task_desc_str=self._task_desc_str,
            cot_example_str=self._cot_example_str,
            problem_format_str=self._problem_format_str,
            reset=False,
            reward_model_fn=self.reward_model_fn,
        )
        env.math_problem = copy.deepcopy(self.math_problem)
        env._legal_actions = copy.deepcopy(self._legal_actions)
        env.action_history = copy.deepcopy(self.action_history)
        env.values = copy.deepcopy(self.values)
        env._init_query = copy.deepcopy(self._init_query)
        env._next_state_terminated = copy.deepcopy(self._next_state_terminated)
        return env

    @property
    def legal_actions(self):
        return self._legal_actions
