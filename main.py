# main.py
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Dataset: {cfg.dataset.path}")
    print(f"Model: {cfg.model.name}")


if __name__ == "__main__":
    main()
