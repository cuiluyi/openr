import argparse
import jsonlines

from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams


def create_dataset(args, tokenizer):
    dataset = load_dataset(
        args.dataset_name,
        name=args.dataset_subset,
        split=args.dataset_split,
    )

    def make_conversation(example):
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": example[args.question_column]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        example["prompt"] = prompt
        return example

    dataset = dataset.map(make_conversation)
    return dataset


def generate(prompt, llm, sampling_params):
    temp = prompt
    for _ in range(30):
        temp += "<｜begin▁of▁system1｜>\n"
        outputs = llm.generate([temp], sampling_params)
        completion = outputs[0].outputs[0].text
        if completion.endswith("<｜end▁of▁sentence｜>"):
            temp += completion
            break
        elif completion.endswith("<｜begin▁of▁system1｜>"):
            completion = completion.removesuffix("<｜begin▁of▁system1｜>")
        elif completion.endswith("<｜begin▁of▁system2｜>"):
            completion = completion.removesuffix("<｜begin▁of▁system2｜>")
        temp += completion
    return temp[len(prompt) :]


def vllm_generate(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    dataset = create_dataset(args, tokenizer)

    llm = LLM(
        model=args.model_name,
        dtype=args.dtype,
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        # use_cache=False,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_output_tokens,
        include_stop_str_in_output=args.include_stop_str_in_output,
        skip_special_tokens=args.skip_special_tokens,
        stop=["<｜begin▁of▁system1｜>", "<｜begin▁of▁system2｜>", "<｜end▁of▁sentence｜>"],
    )

    writer = jsonlines.open(args.output_name, "w")

    total_acc = 0
    for data in dataset:
        gold_answer = data[args.solution_column]
        prompt = data["prompt"]

        # outputs = llm.generate([prompt], sampling_params)
        completion = generate(prompt, llm, sampling_params)

        acc_score = accuracy_reward(completion, gold_answer)
        total_acc = total_acc + acc_score

        writer.write(
            {
                "prompt": prompt,
                "completion": completion,
                "gold answer": gold_answer,
                "acc scores": acc_score,
            }
        )

    print("=" * 100)
    print("eval acc: ", total_acc / len(dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        required=True,
        help="model name path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "auto"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="number of GPUs to use for generation",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization to prevent OOM",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4/MATH-500",
        required=True,
        help="dataset path",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default=None,
        help="dataset subset to use for evaluation",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="dataset split to use for evaluation",
    )
    parser.add_argument(
        "--question_column",
        type=str,
        default="question",
        help="field in the dataset that contains the question",
    )
    parser.add_argument(
        "--solution_column",
        type=str,
        default="solution",
        help="field in the dataset that contains the solution",
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default="Please reason step by step, and put your final answer within \\boxed{}.",
        help="system prompt for the model",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        required=True,
        help="output path",
    )

    # Sampling parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for sampling",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=32768,
        help="Maximum number of output tokens per generation",
    )
    parser.add_argument(
        "--include_stop_str_in_output",
        action="store_true",
        help="Whether to include stop strings in the output",
    )
    parser.add_argument(
        "--skip_special_tokens",
        action="store_true",
        help="Whether to skip special tokens in the output",
    )

    args = parser.parse_args()

    from pprint import pp

    pp(args)

    vllm_generate(args)
