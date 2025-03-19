from argparse import ArgumentParser

from config.config_utils import str2bool

# 1.创建参数解析器
parser = ArgumentParser(description="这是一个解析命令行参数示例")

# 2.添加参数
parser = ArgumentParser()
parser.add_argument("--LM", type=str, required=True)
parser.add_argument("--RM", type=str, default="dummy")
parser.add_argument("--controller_addr", type=str, default="http://0.0.0.0:28778")
# task config
parser.add_argument("--task_name", type=str, default="gsm8k")
parser.add_argument("--test", type=str2bool, default=True)
parser.add_argument("--is_few_shot", type=str2bool, default=False)
parser.add_argument("--seed", type=int, default=0)
# method config
parser.add_argument("--method", type=str, required=True)
parser.add_argument("--num_sequence", type=int, default=1)
# LM gen config
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_k", type=int, default=-1)
parser.add_argument("--top_p", type=float, default=1)
parser.add_argument("--max_new_tokens", type=int, default=256)
# Tree construction config
parser.add_argument("--tree_max_depth", type=int, default=None)
parser.add_argument("--tree_max_width", type=int, default=None)
# ckpg config
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--resume_dir", type=str, default=None)
# parallel config
parser.add_argument("--local", action="store_true", default=False)
parser.add_argument("--num-worker", type=int, default=32)       # 注意这里参数名有 - 符号，后面会转化为 _ 符号
config = parser.parse_args()

# 3.解析命令行参数
args = parser.parse_args()

# 4.获取并打印参数值
print(args)