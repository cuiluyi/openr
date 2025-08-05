import json
import jsonlines
import shutil

from pathlib import Path
from typing import Dict, Union


def append_to_jsonl(file_path: str, data: Dict[str, str]):
    """
    Appends a dictionary as a new line to a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.
        data (dict): Data to append. Must be JSON-serializable.
    """

    with open(file_path, "a", encoding="utf-8") as f:
        json_line = json.dumps(data, ensure_ascii=False)
        f.write(json_line + "\n")


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
    with open(output_file, mode) as outfile:
        for item in data:
            outfile.write(json.dumps(item) + "\n")


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


def copy_file_to_dir(src_file: Union[Path, str], dst_dir: Union[Path, str]) -> None:
    """
    Copy a file to the target directory.

    Args:
        src_file (str): Path to the source file (e.g., 'a/p/c.json').
        dst_dir (str): Path to the target directory (e.g., 'd/q/').

    Returns:
        Path: Path to the copied file in the destination directory.
    """
    src = Path(src_file)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Copy file to destination
    target_file = dst / src.name
    shutil.copy2(src, target_file)  # copy2 preserves metadata
