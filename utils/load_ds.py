import jsonlines
from datasets import load_dataset
from torch.utils.data import Dataset


class JsonlMathDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = []
        with jsonlines.open(data_path, "r") as reader:
            for obj in reader:
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return {"question": x["problem"], "answer": x["solution"]}


def get_train_test_dataset(
    dataset_name: str,
    dataset_split: str,
    dataset_subset: str,
) -> Dataset:
    if dataset_name.endswith((".json", ".jsonl")):
        return JsonlMathDataset(dataset_name)

    dataset = load_dataset(dataset_name, name=dataset_subset, split=dataset_split)

    def map_fields(item):
        return {
            "question": item.get("question") or item.get("problem"),
            "answer": item.get("solution") or item.get("answer"),
        }

    def get_remove_columns():
        original_columns = dataset.column_names
        return [col for col in original_columns if col not in ["question", "answer"]]

    dataset = dataset.map(
        map_fields,
        remove_columns=get_remove_columns(),
    )
    return dataset
