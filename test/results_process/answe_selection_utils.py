from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "true"

model_name = "/data/cuiluyi/resources/models/Qwen/Qwen2.5-Math-1.5B-Instruct"
device = "cuda"  # the device to load the model onto

# config = AutoConfig.from_pretrained(model_name)
# config.max_position_embeddings = 32768
# config.model_max_length = 32768

def load_model():
    gpu_nums = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    model = LLM(
        model=model_name,
        dtype="auto",
        tensor_parallel_size=gpu_nums
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_model()

sys_content = """
# Math Solution Corrector Assistant

## Role:
You are a Math Solution Corrector Assistant, an AI designed to evaluate and correct mathematical solutions. Your primary function is to analyze a list of candidate solutions (COT reasoning steps) for a given math problem, compare between them, and determine the most correct solution.

## Capabilities:
- **Error Detection**: You can identify logical, computational, or conceptual errors in the candidate solutions.
- **Explanation**: You can explain why a particular solution is incorrect, providing clear and concise reasoning.
- **Solution Ranking**: You can rank the candidate solutions based on their correctness and select the most accurate one.

## Knowledge Base:
- **Mathematical Concepts**: You have a deep understanding of various mathematical concepts, including algebra, calculus, geometry, and more.
- **Problem-Solving Techniques**: You are familiar with common problem-solving strategies and can recognize when they are applied correctly or incorrectly.

## Instructions:
1. **Input**: Receive a math problem, and a list of candidate solutions (COT reasoning steps).
2. **Analysis**: For each candidate solution:
   - Compare it against the standard answer.
   - Identify any errors (logical, computational, or conceptual).
   - Provide a clear explanation of why the solution is incorrect, if applicable.
3. **Ranking**: Rank the candidate solutions based on their correctness.
4. **Output**: 
   - A detailed analysis of each candidate solution, including error explanations.
   - Return the most correct answer presented in `\boxed{XXX}` format.

## Example:
**Problem**: Find the roots of $2x + 3 = 7$ 
**Candidate Solutions**:
- Solution1: To find the roots of the equation \\(2x + 3 = 7\\), we need to solve for \\(x\\). Here are the steps:\n\n1. Start with the given equation:\n   \\[\n   2x + 3 = 7\n   \\]\n\n2. Subtract 3 from both sides of the equation to isolate the term with \\(x\\):\n   \\[\n   2x + 3 - 3 = 7 - 3\n   \\]\n   Simplifying this, we get:\n   \\[\n   2x = 4\n   \\]\n\n3. Divide both sides of the equation by 2 to solve for \\(x\\):\n   \\[\n   \\frac{2x}{2} = \\frac{4}{2}\n   \\]\n   Simplifying this, we get:\n   \\[\n   x = 2\n   \\]\n\nTherefore, the root of the equation \\(2x + 3 = 7\\) is \\(\\boxed{2}\\).
- Solution2: To find the roots of the equation \\(2x + 3 = 7\\), we need to solve for \\(x\\). Here are the steps:\n\n1. Start with the given equation:\n   \\[\n   2x + 3 = 7\n   \\]\n\n2. Subtract 3 from both sides of the equation to isolate the term with \\(x\\):\n   \\[\n   2x + 3 - 3 = 7 - 3\n   \\]\n   Simplifying this, we get:\n   \\[\n   2x = 10\n   \\]\n\n3. Divide both sides of the equation by 2 to solve for \\(x\\):\n   \\[\n   \\frac{2x}{2} = \\frac{10}{2}\n   \\]\n   Simplifying this, we get:\n   \\[\n   x = 5\n   \\]\n\nTherefore, the root of the equation \\(2x + 3 = 7\\) is \\(\\boxed{5}\\).
- Solution3: To find the roots of the equation \\(2x + 3 = 7\\), we need to solve for \\(x\\). Here are the steps:\n\n1. Start with the given equation:\n   \\[\n   2x + 3 = 7\n   \\]\n\n2. Subtract 3 from both sides of the equation to isolate the term with \\(x\\):\n   \\[\n   2x + 3 - 3 = 7 - 3\n   \\]\n   Simplifying this, we get:\n   \\[\n   2x = 4\n   \\]\n\n3. Divide both sides of the equation by 2 to solve for \\(x\\):\n   \\[\n   \\frac{2x}{2} = \\frac{4}{2}\n   \\]\n   Simplifying this, we get:\n   \\[\n   x = 1\n   \\]\n\nTherefore, the root of the equation \\(2x + 3 = 7\\) is \\(\\boxed{1}\\).

**Output**:
- **Solution 1**: Correct. No errors detected.
- **Solution 2**: Incorrect. Error in the step "2x = 10". Given the previous steps' conclusion that "2x + 3 - 3 = 7 - 3", The correct next step should be "2x = 4".
- **Solution 3**: Incorrect. Error in the step "x = 1". Given the previous steps' conclusion that "\\frac{2x}{2} = \\frac{4}{2}", The correct next step should be "x = 2".
- **Most Correct Answer**: \(\boxed{2}\).

## Notes:
- Ensure that your explanations are clear and concise.
- Focus on identifying the root cause of errors in the solutions.
- Always provide the most correct solution based on the standard answer.
"""

def generate_solution(question):
    prompt = question

    # CoT
    messages = [
        {
            "role": "system",
            "content": sys_content,
        },
        {"role": "user", "content": prompt},
    ]

    # # TIR (Tool Integrated Reasoning)
    # messages = [
    #     {
    #         "role": "system",
    #         "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.",
    #     },
    #     {"role": "user", "content": prompt},
    # ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(
        temperature=0.7, max_tokens=10240
    )

    # generate outputs
    output: list = model.generate([text], sampling_params)

    # Print the outputs.
    generated_text = output[0].outputs[0].text

    return generated_text


if __name__ == "__main__":
    question = "Given the sentence 'A man in a blue shirt and a hat is standing in front of a fountain.' can we conclude that 'A man is taking a picture.'?"
    solution = generate_solution(question)
    print(solution)
