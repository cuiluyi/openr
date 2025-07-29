import re
import requests

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


def _value_inference_fastchat(
    model_name: str,
    input_str: Union[List[str], str],
    controller_addr="http://0.0.0.0:28777",
):
    ret = requests.post(controller_addr + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]
    if not worker_addr:
        raise ValueError("Value Model name {} does not exist.".format(model_name))

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {"input_str": input_str}
    response = requests.post(
        worker_addr + "/worker_value_inference",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    results = response.json()
    value = results["value"]
    return value


@dataclass
class RewardModelBaseConfig:
    step_tag: str
    # a format string that takes in question and answer
    #  need to have {question} and {answer} in the string
    format_str: str


class RewardModelCallingFunction:
    def __init__(self, config: RewardModelBaseConfig):
        self.config = config
        self.step_tag = config.step_tag
        self.format_str = config.format_str

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        lm_step_tag: Union[str, List[str]],
    ) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    def replace_step_tag(
        self,
        answer: str,
        lm_step_tag: Union[str, List[str]],
    ) -> str:
        if isinstance(lm_step_tag, str):
            splits = answer.split(lm_step_tag)
            splits = [s.strip() for s in splits]
        else:
            assert isinstance(lm_step_tag, list)
            pattern = "|".join(re.escape(d) for d in lm_step_tag)
            splits = re.split(pattern, answer)
        # add a whitespace to avoid tokenization issue
        response = f" {self.step_tag}".join([s for s in splits if s != ""])
        response += f" {self.step_tag}"
        return response


@dataclass
class RemoteRewardModelConfig(RewardModelBaseConfig):
    model_name: str
    controller_addr: str


class RMRemoteCaller(RewardModelCallingFunction):
    def __init__(self, config: RemoteRewardModelConfig):
        self.model_name = config.model_name
        self.controller_addr = config.controller_addr
        super().__init__(config)

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        lm_step_tag: Union[str, List[str]],
    ) -> Union[List[int], List[List[int]]]:

        if isinstance(question_answer_pairs[0], str):
            question, answer = question_answer_pairs
            input_str = self.format_str.format(
                question=question,
                answer=self.replace_step_tag(answer, lm_step_tag),
            )
        else:
            input_str = [
                self.format_str.format(
                    question=question,
                    answer=self.replace_step_tag(answer, lm_step_tag),
                )
                for question, answer in question_answer_pairs
            ]
        return _value_inference_fastchat(
            input_str=input_str,
            model_name=self.model_name,
            controller_addr=self.controller_addr,
        )
