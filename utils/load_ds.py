from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset
from utils import read_jsonl


@dataclass
class DataItem:
    question: Optional[str] = None
    gold: Optional[str] = None


def get_train_test_dataset(
    dataset_name: str,
    dataset_split: str,
    dataset_subset: str,
) -> List[DataItem]:
    # Case 1: Load from a local .json or .jsonl file
    if dataset_name.endswith((".json", ".jsonl")):
        raw_data = read_jsonl(dataset_name)
        return [
            DataItem(
                question=item.get("question") or item.get("problem"),
                gold=item.get("solution") or item.get("answer"),
            )
            for item in raw_data
        ]

    # Case 2: Load from HuggingFace dataset hub
    dataset = load_dataset(dataset_name, name=dataset_subset, split=dataset_split)

    def extract_data_item(item) -> DataItem:
        return DataItem(
            question=item.get("question") or item.get("problem"),
            gold=item.get("solution") or item.get("answer"),
        )

    return [extract_data_item(item) for item in dataset]
