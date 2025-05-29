from math_verify import parse, verify
import jsonlines

pre_jsonl = "/data/cuiluyi/openr/train/efficient/data/process/record.jsonl"
post_jsonl = "/data/cuiluyi/openr/train/efficient/data/process/streamline.jsonl"
save_jsonl = "/data/cuiluyi/openr/train/efficient/data/sft_data.jsonl"


with jsonlines.open(pre_jsonl) as pre_reader, jsonlines.open(
    post_jsonl
) as post_reader, jsonlines.open(save_jsonl, mode="w") as writer:
    for item1, item2 in zip(pre_reader, post_reader):
        groundtruth = parse(item1["groundtruth"])
        raw_answer = parse(item2["original_task"])
        eff_answer = parse(item2["simplified_solution"])

        if verify(groundtruth, raw_answer) and verify(groundtruth, eff_answer):
            writer.write(
                {
                    "prompt": item1["question"],
                    "completion": item2["simplified_solution"],
                }
            )
