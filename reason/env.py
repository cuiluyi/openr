import copy

from typing import Optional, Callable
from math_verify import parse, verify

from utils import extract_answer, softmax
from utils.distributed import print_with_rank
from utils.prompt import build_query_str
from reason.infer.lm_call import LMCallingConfig
from reason.infer.lm_call import ConcatedLMGenResult, merge_concated_results

SYSTEM1_BEGIN_TAG = "<｜begin▁of▁system1｜>\n"
SYSTEM2_BEGIN_TAG = "<｜begin▁of▁system2｜>\n"
SYSTEM1_END_TAG = "\n<｜end▁of▁system1｜>"
SYSTEM2_END_TAG = "\n<｜end▁of▁system2｜>"


class NoLegalActionException(Exception):
    pass


class ResetException(Exception):
    pass


class Env:
    """The basic environment for solving natural language problems using CoT"""

    sep: str = "\n\n"

    def __init__(
        self,
        config,
        math_problems,
        lm_call,
        reset=True,
        rm_call: Optional[Callable] = None,
    ):
        self.config = config
        self.mcts_mode = "play_with_bot_mode"
        self.math_problems = math_problems
        self.lm_call = lm_call
        self.action_history = None
        self.values = []
        self.math_problem = None
        self._legal_actions = None
        self._stop_str = config.get("stop_str", None)

        self.rm_call = rm_call

        if reset:
            self.reset(update_legal_action=True)

    def reset(self, update_legal_action=True):
        self.set_problem(idx=0)  # reset environment to problem idx
        self.action_history = []
        self.values = []
        self._init_query = build_query_str(problem_input=self.math_problem["question"])
        api_completion_token = 0
        if update_legal_action:
            cnt = 0
            while cnt < 5:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = self.update_legal_actions()
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
                    self._legal_actions, api_completion_token = self.update_legal_actions()
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
        ret = self._init_query + "".join(item for item in self.action_history if item is not None)
        return ret

    def update_legal_actions(self):
        fast_try_num = self.config["max_actions"] // 2
        slow_try_num = self.config["max_actions"] - fast_try_num
        # fast generation
        fast_result: ConcatedLMGenResult = self.lm_call(
            input_str=self.get_state() + SYSTEM1_BEGIN_TAG,
            config=LMCallingConfig(
                n=fast_try_num,
                stop_str=SYSTEM1_END_TAG,
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
                stop_str=SYSTEM2_END_TAG,
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
            terminated = not result.text[i].endswith((SYSTEM1_END_TAG, SYSTEM2_END_TAG))
            
            if (
                len(result.text[i]) > 0
                and result.text[i] not in text_list
                and result.finish_reason[i] == "stop"
            ):
                text_list.append(result.text[i])
                logprob_list.append(result.logp_avg_by_len[i])
                num_token_list.append(result.num_tokens[i])
                finish_reason_list.append(result.finish_reason[i])
                next_state_terminated[result.text[i]] = terminated

        if len(logprob_list) == 0:
            print_with_rank("state: {}".format(self.get_state()))
            print_with_rank("gen_result: {}".format(result))
            raise NoLegalActionException("No possible action have been generated.")

        prob_list = softmax(logprob_list)

        _legal_actions = [
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
        self._next_state_terminated = next_state_terminated
        return _legal_actions, result.completion_tokens

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        return verify(self.math_problem["answer"], extracted_answer)

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0

    @property
    def stop_str(self):
        return self._stop_str

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
            lm_call=self.lm_call,
            reset=False,
            rm_call=self.rm_call,
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
