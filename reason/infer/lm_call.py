import requests
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class ConcatedLMGenResult:
    text: List[str]
    prompt_tokens: int
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]
    finish_reason: List[str]

    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)

    def __len__(self):
        return len(self.text)


def merge_concated_results(results: List[ConcatedLMGenResult]) -> ConcatedLMGenResult:
    text = []
    prompt_tokens = []
    num_tokens = []
    cumulative_logprob = []
    logp_avg_by_len = []
    finish_reason = []

    for result in results:
        text.extend(result.text)
        prompt_tokens.append(result.prompt_tokens)
        num_tokens.extend(result.num_tokens)
        cumulative_logprob.extend(result.cumulative_logprob)
        logp_avg_by_len.extend(result.logp_avg_by_len)
        finish_reason.extend(result.finish_reason)

    return ConcatedLMGenResult(
        text=text,
        prompt_tokens=sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0,
        num_tokens=num_tokens,
        cumulative_logprob=cumulative_logprob,
        logp_avg_by_len=logp_avg_by_len,
        finish_reason=finish_reason,
    )


def _generate_fastchat(
    query_str: str,
    model_name: str,
    n: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    stop_token_ids: Optional[List[int]] = None,
    stop_str: Optional[List[str]] = None,
    include_stop_str_in_output: bool = False,
    controller_addr: str = "http://0.0.0.0:28777",
    seed: Optional[int] = None,
) -> ConcatedLMGenResult:

    headers = {"User-Agent": "FastChat Client"}
    ret = requests.post(controller_addr + "/get_worker_address", json={"model": model_name})
    worker_addr = ret.json()["address"]
    if not worker_addr:
        raise ValueError("Language Model name {} does not exist.".format(model_name))

    gen_params = {
        "model": model_name,
        "prompt": query_str,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "top_k": top_k,
        "stop_token_ids": stop_token_ids,
        "max_new_tokens": max_new_tokens,
        "stop": stop_str,
        "echo": False,
        "include_stop_str_in_output": include_stop_str_in_output,
        "seed": seed,
    }

    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    results = response.json()

    output_token_lens = results["output_token_len"]
    cum_logps = results["cumulative_logprob"]
    avg_len_logps = [clp / max(1, otl) for clp, otl in zip(cum_logps, output_token_lens)]

    return ConcatedLMGenResult(
        text=results["text"],
        prompt_tokens=results["usage"]["prompt_tokens"],
        num_tokens=results["output_token_len"],
        cumulative_logprob=cum_logps,
        logp_avg_by_len=avg_len_logps,
        finish_reason=results["finish_reason"],
    )


@dataclass
class LMCallingConfig:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 512
    stop_token_ids: Optional[List[int]] = None
    stop_str: Optional[Union[str, List[str]]] = None
    include_stop_str_in_output: bool = False
    seed: Optional[int] = None


class LanguageModelCallingFunction:
    def __init__(self, lm_step_tag: Union[str, List[str]] = None):
        self.lm_step_tag = lm_step_tag

    def __call__(self, input_str: str, config: LMCallingConfig) -> ConcatedLMGenResult:
        raise NotImplementedError


class VLLMRemoteCaller(LanguageModelCallingFunction):
    def __init__(
        self,
        model_name: str,
        controller_addr: str = "http://0.0.0.0:28777",
        lm_step_tag: Union[str, List[str]] = None,
    ):
        self.model_name = model_name
        self.controller_addr = controller_addr
        super().__init__(lm_step_tag)

    def __call__(self, input_str: str, config: LMCallingConfig) -> ConcatedLMGenResult:
        # FastChat implementation, batch generation
        return _generate_fastchat(
            query_str=input_str,
            model_name=self.model_name,
            n=config.n,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_new_tokens=config.max_new_tokens,
            stop_token_ids=config.stop_token_ids,
            stop_str=config.stop_str,
            controller_addr=self.controller_addr,
            include_stop_str_in_output=config.include_stop_str_in_output,
            seed=config.seed,
        )
