from pathlib import Path
import jsonlines
from torch.utils.data import Dataset


def get_train_test_dataset(*args, **kwargs):
    env_dir = Path(__file__).parent
    test_ds = JsonlMathDataset(env_dir / "dataset/test500.jsonl")
    # test_ds = JsonlMathDataset(env_dir / "dataset/test_demo.jsonl")
    # test_ds = JsonlMathDataset("/data/cuiluyi/openr/results/MATH/vanila_mcts/20250224_122252/error_data.jsonl")
    # test_ds = JsonlMathDataset("/data/cuiluyi/openr/results/MATH/vanila_mcts/20250224_122252/incorrect_data.jsonl")
    train_ds = JsonlMathDataset(env_dir / "dataset/train.jsonl")
    return train_ds, test_ds


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
