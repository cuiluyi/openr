from .seed import setup_seed

from .parse_answer import extract_answer, accuracy_reward

from .file_io import read_jsonl, write_jsonl, jsonl_to_json


def str2bool(x: str):
    if x == "False":
        return False
    elif x == "True":
        return True
    else:
        raise ValueError(f'you should either input "True" or "False" but not {x}')
