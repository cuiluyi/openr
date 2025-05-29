import time

import gradio as gr
from gradio import Interface, Dropdown, Textbox, Button
import json
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RewardModelBaseConfig,
    RemoteRewardModelConfig,
)
from reason.evaluation.evaluator import SolutionOutput, Task, RemoteMathEvaluator
from reason.evaluation.methods import *
from reason.reranking.vote_utils import (
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_VOTE,
    PRM_LAST_MAX,
    AGG_FN_MAP,
)


def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("### Question")
        question = gr.Textbox(label="Enter your question")

        gr.Markdown("### Decoding Parameters")
        with gr.Row():
            with gr.Column():
                temperature = gr.Number(
                    label="Temperature",
                    value=1.0,
                )
                top_p = gr.Number(
                    label="Top P",
                    value=0.95,
                )
            with gr.Column():
                max_new_tokens = gr.Number(
                    label="Max New Tokens",
                    value=2048,
                    precision=0,
                )
                top_k = gr.Number(
                    label="Top K",
                    value=-1,
                    precision=0,
                )

        gr.Markdown("### Script Parameters")
        with gr.Row():
            with gr.Column():
                save_dir = gr.Textbox(
                    label="Save Directory",
                    value="./results",
                )
                resum_dir = gr.Textbox(
                    label="Resume Directory (optional)",
                )
                seed = gr.Number(
                    label="Seed",
                    value=42,
                    precision=0,
                )
            with gr.Column():
                num_worker = gr.Number(
                    label="Number of Workers",
                    value=4,
                    precision=0,
                )
                controller_addr = gr.Textbox(
                    label="Controller Address",
                    value="http://0.0.0.0:28777",
                )

        gr.Markdown("### Search Algorithm Parameters")
        method = gr.Dropdown(
            choices=[
                "Cot",
                "Best-of-N",
                "Beam-Search",
                "Vanilla-MCTS",
                "Alpha-MCTS",
                "MCTS-Beam",
            ],
            label="Select Structured-Search Algorithm",
            value="Cot",
        )
        with gr.Column():
            with gr.Row():
                policy_model = gr.Textbox(
                    label="Language Model Name",
                    value="Qwen2.5-Math-1.5B-Instruct",
                )
                reward_model = gr.Textbox(
                    label="Reward Model Name",
                    value="Qwen2.5-Math-PRM-7B",
                )
            with gr.Row():
                tree_max_width = gr.Number(
                    label="Tree Max Width",
                    value=4,
                    precision=0,
                )
                tree_max_depth = gr.Number(
                    label="Tree Max Depth",
                    value=50,
                    precision=0,
                )
            with gr.Row():
                num_sequence = gr.Number(
                    label="Number of Sequences",
                    value=1,
                    precision=0,
                )
                simulate_num = gr.Number(
                    label="Simulate Number",
                    value=4,
                    precision=0,
                )

        gr.Markdown("### Solution Selector Parameters")
        aggration_mode = gr.Dropdown(
            choices=[
                MAJORITY_VOTE,
                PRM_MIN_MAX,
                PRM_MIN_VOTE,
                PRM_LAST_VOTE,
                PRM_LAST_MAX,
            ],
            label="Select Solution Selector Algorithm",
            value=MAJORITY_VOTE,
        )

        submit_btn = gr.Button("Submit")

        gr.Markdown("### Output")
        output = gr.Textbox(
            label="All Solutions",
            interactive=False,
        )
        result = gr.Textbox(
            label="Final Solution",
            interactive=False,
        )

        # Handle form submission
        def handle_submit(*params):
            # Unpack parameters
            (
                question,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                save_dir,
                resum_dir,
                seed,
                num_worker,
                controller_addr,
                method,
                policy_model,
                reward_model,
                tree_max_width,
                tree_max_depth,
                num_sequence,
                simulate_num,
                aggration_mode,
            ) = params

            if "mistral" in reward_model.lower():
                prm_step_tag = "ки\n"
                prm_format_str = "{question} {answer}"
            elif "qwen2.5-math-prm" in reward_model.lower():
                prm_step_tag = "<extra_0>"
                prm_format_str = (
                    "{question}<this is qwen2.5-math-prm seperation &&&&& >{answer}"
                )
            else:
                # assume qwen
                prm_step_tag = "\n\n\n\n\n "
                prm_format_str = "{question} {answer}"

            if "mistral" in policy_model.lower():
                lm_step_tag = "ки\n"
            else:
                # assume qwen
                lm_step_tag = "\n\n"

            lm_call = VLLMRemoteCaller(
                model_name=policy_model,
                controller_addr=controller_addr,
                lm_step_tag=lm_step_tag,
            )

            rm_config = RemoteRewardModelConfig(
                step_tag=prm_step_tag,
                format_str=prm_format_str,
                model_name=reward_model,
                controller_addr=controller_addr,
            )
            rm_call = RMRemoteCaller(rm_config)

            task = Task(
                task_name="Online",
                dataset_id=None,
            )

            gen_config = LMCallingConfig(
                n=num_sequence,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )

            problem_inst = {
                "question": question,
                "answer": "2",
            }
            method = method.replace("-", "_")
            if method.lower() == "cot":
                method_config = CoTConfig(
                    task_name="Online",
                )
                output: TreeSearchSolutionOutput = cot(
                    config=method_config,
                    gen_config=gen_config,
                    task=task,
                    problem_inst=problem_inst,
                    lm_call=lm_call,
                    rm_call=rm_call,
                )
            elif method.lower() == "best_of_n":
                method_config = BestOfNConfig(
                    task_name="Online",
                    num_sequence=num_sequence,
                )
                output: TreeSearchSolutionOutput = best_of_n(
                    config=method_config,
                    gen_config=gen_config,
                    task=task,
                    problem_inst=problem_inst,
                    lm_call=lm_call,
                    rm_call=rm_call,
                )
            elif method.lower() == "beam_search":
                method_config = BeamSearchConfig(
                    task_name="Online",
                    tree_max_depth=tree_max_depth,
                    tree_max_width=tree_max_width,
                    beam_size=num_sequence,
                )
                output: TreeSearchSolutionOutput = beam_search(
                    config=method_config,
                    gen_config=gen_config,
                    task=task,
                    problem_inst=problem_inst,
                    lm_call=lm_call,
                    rm_call=rm_call,
                )
            elif method.lower() == "vanilla_mcts":
                method_config = VanilaMCTSConfig(
                    task_name="Online",
                    tree_max_width=tree_max_width,
                    tree_max_depth=tree_max_depth,
                    select_by_prior=False,
                    num_path=num_sequence,
                )
                output: TreeSearchSolutionOutput = vanila_mcts(
                    config=method_config,
                    gen_config=gen_config,
                    task=task,
                    problem_inst=problem_inst,
                    lm_call=lm_call,
                    rm_call=rm_call,
                )
            elif method.lower() == "alpha_mcts":
                method_config = MCTSConfig(
                    task_name="Online",
                    tree_max_width=tree_max_width,
                    tree_max_depth=tree_max_depth,
                    select_by_prior=False,
                    num_path=num_sequence,
                    simulate_num=simulate_num,
                )
                output: TreeSearchSolutionOutput = mcts(
                    config=method_config,
                    gen_config=gen_config,
                    task=task,
                    problem_inst=problem_inst,
                    lm_call=lm_call,
                    rm_call=rm_call,
                )
            elif method.lower() == "mcts_beam":
                method_config = MCTSBeamSearchConfig(
                    task_name="Online",
                    tree_max_width=tree_max_width,
                    tree_max_depth=tree_max_depth,
                    select_by_prior=False,
                    num_path=num_sequence,
                    simulate_num=simulate_num,
                )
                output: TreeSearchSolutionOutput = mcts_beam_search(
                    config=method_config,
                    gen_config=gen_config,
                    task=task,
                    problem_inst=problem_inst,
                    lm_call=lm_call,
                    rm_call=rm_call,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            gen_answers = output.solutions
            input_list = [(problem_inst["question"], txt) for txt in gen_answers]
            value_list = rm_call(input_list, lm_step_tag=lm_call.lm_step_tag)

            ans_list = [parse(txt) for txt in gen_answers]
            aggregated_ans = AGG_FN_MAP[aggration_mode](ans_list, value_list)

            for i, ans in enumerate(ans_list):
                if verify(aggregated_ans, ans):
                    final_solution = gen_answers[i]
                    break

            all_solutions = ""
            for i, item in enumerate(gen_answers):
                all_solutions += f"Solution{i + 1}:\n\n {item}\n"

            return all_solutions, final_solution

        submit_btn.click(
            handle_submit,
            inputs=[
                question,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                save_dir,
                resum_dir,
                seed,
                num_worker,
                controller_addr,
                method,
                policy_model,
                reward_model,
                tree_max_width,
                tree_max_depth,
                num_sequence,
                simulate_num,
                aggration_mode,
            ],
            outputs=[
                output,
                result,
            ],
        )

    return interface


if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)
