from transformers import AutoTokenizer

tokenizer_path = "/data/cuiluyi/openr/ckpts/DeepSeek-R1-Distill-Qwen-1.5B/slow_fast_reason-sft-s1k-1.1_full/checkpoint-400"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def get_tokens_num(text: str) -> int:
    """Get the number of tokens in a given text using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


if __name__ == "__main__":
    text = "<｜begin▁of▁system2｜>\n<think>Okay, so I have this equation here: 1 over x minus 1 over y equals 1 over z. And I need to solve for z. Let me see. Alright, let me write that down again to make sure I have it right: (1/x) - (1/y) = 1/z. Hmm, okay. I need to manipulate this equation to get z by itself. First, maybe I can combine the fractions on the left side. To subtract these, I need a common denominator. The denominators are x and y, so the least common denominator would be xy. So I'll rewrite each fraction: 1/x is the same as y/(xy), and 1/y is x/(xy). So subtracting them gives y/(xy) - x/(xy) = (y - x)/xy. So the left side simplifies to (y - x)/xy. And that's equal to 1/z. So (y - x)/xy = 1/z.</think>\nTo solve for \\( z \\) in the equation \\(\\frac{1}{x} - \\frac{1}{y} = \\frac{1}{z}\\), we start by combining the fractions on the left side.\n<｜end▁of▁system2｜><｜begin▁of▁system1｜>\nFirst, find a common denominator for the fractions on the left:\n\\[\n\\frac{1}{x} - \\frac{1}{y} = \\frac{y - x}{xy}\n\\]\n<｜end▁of▁system1｜><｜begin▁of▁system1｜>\nThis gives us the equation:\n\\[\n\\frac{y - x}{xy} = \\frac{1}{z}\n\\]\n<｜end▁of▁system1｜><｜begin▁of▁system1｜>\nCross-multiplying to solve for \\( z \\):\n\\[\nz(y - x) = xy\n\\]\n<｜end▁of▁system1｜><｜begin▁of▁system1｜>\nDividing both sides by \\( y - x \\):\n\\[\nz = \\frac{xy}{y - x}\n\\]\n<｜end▁of▁system1｜><｜begin▁of▁system1｜>\nThus, the correct answer is \\(\\boxed{D}\\).\n<｜end▁of▁system1｜><｜end▁of▁sentence｜>"
    print(get_tokens_num("/data/cuiluyi/resources/models/Qwen/Qwen2.5-1.5B-Instruct", text))
