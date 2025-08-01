from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--LM", type=str, required=True)
    parser.add_argument("--RM", type=str, default="dummy")
    parser.add_argument("--controller_addr", type=str, default="http://0.0.0.0:28777")
    # dataset config
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--subset", type=str, default=None)
    # method config
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--num_sequence", type=int, default=1)
    parser.add_argument("--simulate_num", type=int, default=1)
    # LM gen config
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--stop_str", type=str, default=None)
    # Tree construction config
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_width", type=int, default=None)
    # ckpg config
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--resume_dir", type=str, default=None)
    # parallel config
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--num_worker", type=int, default=32)

    args = parser.parse_args()
    return args
