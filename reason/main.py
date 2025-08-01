from datetime import datetime
from functools import partial
from pathlib import Path

import ray
from loguru import logger
from reason.infer.methods import (
    BeamSearchConfig,
    VanillaMCTSConfig,
    beam_search,
    vanilla_mcts,
)
from reason.infer.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.infer.rm_call import RMRemoteCaller, RemoteRewardModelConfig
from reason.config import parse_args
from reason.eval.evaluate import parallel_evaluate_dataset
from utils import setup_seed, write_json
from utils import get_train_test_dataset


args = parse_args()
if args.seed is not None:
    setup_seed(args.seed)

if args.local:
    logger.info("run in pure local mode for debug only")
    args.num_worker = 1
    ray.init(local_mode=True)

cfg_dict_record = dict()

# setup reward model caller
if "mistral" in args.RM.lower():
    prm_step_tag = "ки\n"
    prm_format_str = "{question} {answer}"
elif "qwen" in args.RM.lower():
    prm_step_tag = "<extra_0>"
    prm_format_str = "{question}<｜question▁answer▁delimiter｜>{answer}"
else:
    raise ValueError("Unsupported RM model type: {}".format(args.RM))
rm_config = RemoteRewardModelConfig(
    step_tag=prm_step_tag,
    format_str=prm_format_str,
    model_name=args.RM,
    controller_addr=args.controller_addr,
)
rm_call = RMRemoteCaller(rm_config)
cfg_dict_record["RM"] = args.RM

# setup language model caller
if "mistral" in args.LM.lower():
    lm_step_tag = "ки\n"
elif "qwen" in args.LM.lower():
    lm_step_tag = "\n\n"
elif "checkpoint" in args.LM.lower():
    lm_step_tag = ["<｜end▁of▁system1｜>", "<｜end▁of▁system2｜>"]
else:
    raise ValueError("Unsupported LM model type: {}".format(args.LM))
lm_call = VLLMRemoteCaller(
    args.LM,
    args.controller_addr,
    lm_step_tag=lm_step_tag,
)
cfg_dict_record["LM"] = args.LM

# setup generation config
gen_config = LMCallingConfig(
    n=args.num_sequence,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
    max_new_tokens=args.max_new_tokens,
    seed=args.seed,
    stop_str=args.stop_str,
)
cfg_dict_record["gen_config"] = gen_config.__dict__

# setup method config and solver function
if args.method == "beam_search":
    method_config = BeamSearchConfig(
        max_steps=args.max_steps,
        max_width=args.max_width,
        beam_size=args.num_sequence,
    )
    solver_fn = partial(beam_search, method_config, gen_config)
elif args.method == "vanilla_mcts":
    method_config = VanillaMCTSConfig(
        max_width=args.max_width,
        max_steps=args.max_steps,
        num_path=args.num_sequence,
    )
    solver_fn = partial(vanilla_mcts, method_config, gen_config)
else:
    raise ValueError(f"Unknown method: {args.method}")
cfg_dict_record["method"] = args.method
cfg_dict_record["method_config"] = method_config.__dict__

# load dataset
dataset = get_train_test_dataset(args.dataset, dataset_split=args.split, dataset_subset=args.subset)

# setup save directory and record writer
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path(args.save_dir) / args.method / args.dataset.split("/")[-1] / datetime_str
save_dir.mkdir(parents=True)
write_json(cfg_dict_record, save_dir / "config.json")

parallel_evaluate_dataset(
    solver_fn=solver_fn,
    rm_call=rm_call,
    lm_call=lm_call,
    dataset=dataset,
    save_dir=save_dir,
    resume_dir=Path(args.resume_dir) if args.resume_dir else None,
    num_worker=args.num_worker,
)
