import json
import jsonlines
import shutil

from pathlib import Path
from typing import Dict, List, Union


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


# ensure the last line of a file ends with a newline character
def ensure_newline(file_path: Path) -> None:
    if file_path.exists() and file_path.stat().st_size > 0:
        with open(file_path, "rb+") as f:
            f.seek(-1, 2)  # move to the last byte
            last_char = f.read(1)
            if last_char != b"\n":
                f.write(b"\n")  # append missing newline character


def merge_jsonl_files(input_files: List[Union[str, Path]], output_file: Union[str, Path]) -> None:
    """
    Merge multiple JSONL files into one JSONL file.

    Args:
        input_files (List[Union[str, Path]]): List of input JSONL file paths.
        output_file (Union[str, Path]): Path to the output JSONL file.
    """
    output_path = Path(output_file)

    with output_path.open("w", encoding="utf-8") as fout:
        for file_path in input_files:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"Warning: file {file_path} not found, skipped.")
                continue

            with file_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        # Ensure it's valid JSON before writing
                        json.loads(line)
                        fout.write(line + "\n")
                    except json.JSONDecodeError:
                        print(f"Warning: invalid JSON line in {file_path}, skipped.")
    print(f"Merged {len(input_files)} files into {output_file}")
