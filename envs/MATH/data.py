from pathlib import Path

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


def get_train_test_dataset(dataset_id, *args, **kwargs):

    try:
        dataset = load_dataset(dataset_id)

        def map_fields(example):
            return {
                "question": example.get("question")
                or example["problem"],  # 优先用question字段，不存在则用problem
                "answer": example["solution"],
            }

        def get_remove_columns(split):
            original_columns = dataset[split].column_names
            return [
                col for col in original_columns if col not in ["question", "answer"]
            ]

        # 应用映射并清理字段
        train_ds, test_ds = None, None
        if "train" in dataset:
            train_ds = dataset["train"].map(
                map_fields,
                remove_columns=get_remove_columns("train"),  # 动态计算需要移除的列
                batched=True,  # 加速处理
            )

        if "test" in dataset:
            test_ds = dataset["test"].map(
                map_fields, remove_columns=get_remove_columns("test"), batched=True
            )
        return train_ds, test_ds
    except Exception as e:
        env_dir = Path(__file__).parent
        # test_ds = JsonlMathDataset(env_dir / "dataset/train.jsonl")
        test_ds = JsonlMathDataset(env_dir / "dataset/debug.jsonl")
        # test_ds = JsonlMathDataset("/data/cuiluyi/openr/results/MATH/vanila_mcts/20250224_122252/error_data.jsonl")
        # test_ds = JsonlMathDataset("/data/cuiluyi/openr/results/MATH/vanila_mcts/20250224_122252/incorrect_data.jsonl")
        train_ds = JsonlMathDataset(env_dir / "dataset/train.jsonl")
        return train_ds, test_ds
