import copy

from typing import Optional, Callable, Dict, Any
from math_verify import parse, verify

from utils import softmax, DataItem
from utils.parse_answer import extract_answer, verify_answer
from utils.distributed import print_with_rank
from utils.prompt import build_query_str
from utils.text_processe import (
    has_valid_content,
    split_string,
    SYSTEM1_BEGIN_TAG,
    SYSTEM2_BEGIN_TAG,
    SYSTEM1_END_TAG,
    SYSTEM2_END_TAG,
    SUFFIX,
)
from reason.infer.lm_call import LMCallingConfig
from reason.infer.lm_call import ConcatedLMGenResult, merge_concated_results

MAX_RETRIES = 5


class NoLegalActionException(Exception):
    pass


class ResetException(Exception):
    pass


class Env:
    """The basic environment for solving natural language problems using CoT"""

    sep: str = "\n\n"

    def __init__(
        self,
        config: Dict[str, Any],
        input_inst: DataItem,
        lm_call: Callable,
        reset: bool = True,
        rm_call: Optional[Callable] = None,
    ):
        self.config = config
        self.lm_call = lm_call
        self.rm_call = rm_call
        self.input_inst = input_inst
        self.reason_finished = False
        self.legal_actions = None
        self.action_history = None
        self.values = []
        self.stop_str = config.get("stop_str", None)

        if reset:
            self.reset(update_legal_action=True)

    def reset(self, update_legal_action=True):
        self.values = []
        self.action_history = []
        self.init_query = build_query_str(problem_input=self.input_inst.question)

        if update_legal_action:
            for try_id in range(MAX_RETRIES):
                try:
                    self.legal_actions, api_completion_token = self.update_legal_actions()
                    break
                except NoLegalActionException:
                    if try_id == MAX_RETRIES - 1:
                        raise ResetException

        return api_completion_token

    def step(self, action, value=0, update_legal_action=True):
        self.action_history.append(action)
        self.values.append(value)

        # update legal actions
        self.update_reason_finished()

        if not self.reason_finished and update_legal_action:
            for try_id in range(MAX_RETRIES):
                try:
                    self.legal_actions, api_completion_token = self.update_legal_actions()
                    break
                except NoLegalActionException:
                    if try_id == MAX_RETRIES - 1:
                        self.reason_finished = True
                        self.legal_actions = None
                        api_completion_token = 0

        else:
            self.legal_actions = None
            api_completion_token = 0
        return api_completion_token

    def get_state(self):
        # not join about sep_str here because we let vllm return with sep_str
        ret = self.init_query + "".join(item for item in self.action_history if item is not None)
        return ret

    def update_legal_actions(self):
        fast_try_num = self.config["max_width"] // 2
        slow_try_num = self.config["max_width"] - fast_try_num
        # fast generation
        fast_result: ConcatedLMGenResult = self.lm_call(
            input_str=self.get_state() + SYSTEM1_BEGIN_TAG,
            config=LMCallingConfig(
                n=fast_try_num,
                stop_str=[SYSTEM1_BEGIN_TAG, SYSTEM2_BEGIN_TAG],
                # stop_str=self.stop_str,
                include_stop_str_in_output=True,
                **self.config["generation_config"]
            ),
        )
        fast_result.text = [SYSTEM1_BEGIN_TAG + text for text in fast_result.text]

        # slow generation
        slow_result: ConcatedLMGenResult = self.lm_call(
            input_str=self.get_state() + SYSTEM2_BEGIN_TAG,
            config=LMCallingConfig(
                n=slow_try_num,
                stop_str=[SYSTEM1_BEGIN_TAG, SYSTEM2_BEGIN_TAG],
                # stop_str=self.stop_str,
                include_stop_str_in_output=True,
                **self.config["generation_config"]
            ),
        )
        slow_result.text = [SYSTEM2_BEGIN_TAG + text for text in slow_result.text]

        result = merge_concated_results([fast_result, slow_result])

        # process the result
        text_list, logprob_list, num_token_list, finish_reason_list = [], [], [], []
        next_state_terminated = {}

        for i in range(len(result)):
            # XXX: this process can be improve or moved to other place
            # this is a pre-judge of terminal flag or certain action, by
            # whether the text-generation is stop by the <eos> or stop_str
            if result.finish_reason[i] != "stop" or len(result.text[i]) == 0:
                continue

            processed_text = self.post_process_act(result.text[i])
            if not has_valid_content(processed_text):
                continue

            if processed_text not in text_list:
                text_list.append(processed_text)
                logprob_list.append(result.logp_avg_by_len[i])
                num_token_list.append(result.num_tokens[i])
                finish_reason_list.append(result.finish_reason[i])
                next_state_terminated[processed_text] = processed_text.endswith(SUFFIX)

        if len(logprob_list) == 0:
            # print_with_rank("state: {}".format(self.get_state()))
            # print_with_rank("gen_result: {}".format(result))
            raise NoLegalActionException("No possible action have been generated.")

        prob_list = softmax(logprob_list)

        legal_actions = [
            {
                "action": action,
                "prob": prob,
                "num_token": n_token,
                "finish_reason": finish_reason,
            }
            for action, prob, n_token, finish_reason in zip(
                text_list,
                prob_list,
                num_token_list,
                finish_reason_list,
            )
        ]
        self.next_state_terminated = next_state_terminated
        return legal_actions, result.completion_tokens

    def post_process_act(self, action: str):
        step, sep, other = split_string(action, [SYSTEM1_END_TAG, SYSTEM2_END_TAG])
        if action.startswith(SYSTEM1_BEGIN_TAG):
            if sep:
                action = step + SYSTEM1_END_TAG
            else:
                if action.endswith(SUFFIX):
                    return action.removesuffix(SUFFIX) + SYSTEM1_END_TAG + SUFFIX
                else:
                    action = action.removesuffix(SYSTEM1_BEGIN_TAG)
                    action = action.removesuffix(SYSTEM2_BEGIN_TAG)
                    return action + SYSTEM1_END_TAG
        else:
            assert action.startswith(SYSTEM2_BEGIN_TAG)
            if sep:
                action = step + SYSTEM2_END_TAG
            else:
                if action.endswith(SUFFIX):
                    return action.removesuffix(SUFFIX) + SYSTEM2_END_TAG + SUFFIX
                else:
                    action = action.removesuffix(SYSTEM1_BEGIN_TAG)
                    action = action.removesuffix(SYSTEM2_BEGIN_TAG)
                    return action + SYSTEM2_END_TAG

        if other.strip() == SUFFIX:
            action = action + SUFFIX
        return action

    @property
    def question(self) -> str:
        return self.input_inst.question

    @property
    def gold(self) -> str:
        return self.input_inst.gold

    @property
    def answer(self):
        return "".join(self.action_history)

    def update_reason_finished(self):
        assert self.action_history, "action_history should not be empty"
        if self.stop_str is not None and self.stop_str in self.action_history[-1]:
            terminated = True
        elif self.next_state_terminated[self.action_history[-1]]:
            terminated = True
        else:
            terminated = False

        # check if the current state is truncated
        truncated = len(self.action_history) >= self.config["max_steps"]
        self.reason_finished = terminated or truncated

    def copy(self):
        env = self.__class__(
            config=self.config,
            lm_call=self.lm_call,
            input_inst=self.input_inst,
            reset=False,
            rm_call=self.rm_call,
        )
        env.action_history = copy.deepcopy(self.action_history)
        env.values = copy.deepcopy(self.values)
        env.reason_finished = copy.deepcopy(self.reason_finished)
        env.legal_actions = copy.deepcopy(self.legal_actions)
        env.init_query = copy.deepcopy(self.init_query)
        env.next_state_terminated = copy.deepcopy(self.next_state_terminated)
        return env
