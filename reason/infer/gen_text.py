from typing import List, Optional
import requests
from dataclasses import dataclass


@dataclass
class ConcatedLMGenResult:
    text: List[str]
    prompt_tokens: List[int]
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]
    finish_reason: List[str]

    # post init compute number of completion_tokens
    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)


def get_resonse(worker_addr, headers, gen_params):
    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    # print(response)
    results = response.json()
    return results


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
    seed: Optional[int] = 42,
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
    results = get_resonse(worker_addr, headers, gen_params)

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
