from dotenv import load_dotenv
import os

load_dotenv()

# utils/parse_answer.py
# API_KEY = os.getenv("API_KEY")
# BASE_URL = "https://integrate.api.nvidia.com/v1"
# MODEL_NAME = "meta/llama-3.3-70b-instruct"


API_KEY = os.getenv("PAID_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"

TEMPLATE = """You are a mathematical answer validator. You will be provided with a mathematical problem and you need to compare the answer in the reference solution, and the final answer in a model's solution to determine if they are equivalent, even if formatted differently.

PROBLEM:

{problem}

REFERENCE SOLUTION:

{answer}

MODEL'S SOLUTION:

{generation}

Focus ONLY on comparing the final mathematical answer provided by the model while ignoring differences in:

- Formatting (e.g., \\boxed{{}} vs plain text)
- Multiple choice formatting (e.g., "A" vs full solution)
- Order of coordinate pairs or solutions
- Equivalent mathematical expressions or notation variations
- If the model's answer is nonsense, return "Verdict: AMBIGUOUS"

Start with a brief explanation of your comparison (2-3 sentences). Then output your final answer in one of the following formats:

- "Verdict: EQUIVALENT"
- "Verdict: DIFFERENT"
- "Verdict: AMBIGUOUS"
"""
RIGHT_TAG = "Verdict: EQUIVALENT"


DEFAULT_TOKENIZER_PATH = "/data/cuiluyi/resources/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"