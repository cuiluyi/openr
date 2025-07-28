import json
import jsonlines


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def write_jsonl(data, output_file, mode="w"):
    cnt = 0
    with open(output_file, mode) as outfile:
        for item in data:
            outfile.write(json.dumps(item) + "\n")
            cnt += len(item["answer"])
        print("Write {} items into {}".format(cnt, output_file))


def jsonl_to_json(jsonl_file, json_file):
    with jsonlines.open(jsonl_file, "r") as reader:
        data = [obj for obj in reader]

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def json_to_jsonl(json_file, jsonl_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with jsonlines.open(jsonl_file, "w") as writer:
        if isinstance(data, list):
            for item in data:
                writer.write(item)
        else:
            writer.write(data)
