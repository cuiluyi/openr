from typing import List, Optional
import requests
from dataclasses import dataclass
from reason.inference.utils import merge_dict_list


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
    query_str,
    model_name,
    n,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    stop_token_ids,
    stop_str,
    include_stop_str_in_output,
    controller_addr,
    seed,
) -> ConcatedLMGenResult:

    headers = {"User-Agent": "FastChat Client"}
    model_list = model_name.split("&")

    if len(model_list) == 1:
        ret = requests.post(
            controller_addr + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        if not worker_addr:
            raise ValueError(
                "Language Model name {} does not exist.".format(model_name)
            )

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
    else:
        for i in range(n):
            model_name = model_list[i % len(model_list)]
            ret = requests.post(
                controller_addr + "/get_worker_address", json={"model": model_name}
            )
            worker_addr = ret.json()["address"]
            if not worker_addr:
                raise ValueError(
                    "Language Model name {} does not exist.".format(model_name)
                )

            gen_params = {
                "model": model_name,
                "prompt": query_str,
                "temperature": temperature,
                "n": 1,
                "top_p": top_p,
                "top_k": top_k,
                "stop_token_ids": stop_token_ids,
                "max_new_tokens": max_new_tokens,
                "stop": stop_str,
                "echo": False,
                "include_stop_str_in_output": include_stop_str_in_output,
                "seed": seed,
            }
            if i == 0:
                results = get_resonse(worker_addr, headers, gen_params)
            else:
                results = merge_dict_list(
                    [results, get_resonse(worker_addr, headers, gen_params)]
                )

    output_token_lens = results["output_token_len"]
    cum_logps = results["cumulative_logprob"]
    avg_len_logps = [
        clp / max(1, otl) for clp, otl in zip(cum_logps, output_token_lens)
    ]
    # return results["text"], avg_len_logps
    return ConcatedLMGenResult(
        text=results["text"],
        prompt_tokens=results["usage"]["prompt_tokens"],
        num_tokens=results["output_token_len"],
        cumulative_logprob=cum_logps,
        logp_avg_by_len=avg_len_logps,
        finish_reason=results["finish_reason"],
    )
