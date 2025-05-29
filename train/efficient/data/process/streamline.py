from openai import OpenAI
import jsonlines

from tqdm import tqdm
import time  # 用于添加请求之间的延迟

client = OpenAI(
    api_key="sk-3RToo3Zamp9YbXGGVrtAJto1frpSccGxQSgUwShfM29KCEXW",
    base_url="https://api.chatanywhere.tech/v1"
)

# 定义固定的提示前缀
prompt_prefix = """Simplify the following math solution by deleting redundant content.\n\n**Requirements:**  \n1. Keep coherence indicators such as *“First”*, *“Second”*, *“Next”*, *“Finally”*, etc.  \n2. Do **not** output any introductory or concluding statements unrelated to the solution itself.  \n3. Only output the simplified version of the solution, with redundant explanations and repetitions removed.  \n4. Keep necessary mathematical steps, key terms, and equations.\n\n---\n\n**Example:**  \n**Input:** To determine the smallest possible value of \\( c \\) for the function \\( y = a \\sin (bx + c) + d \\), we need to analyze the given graph and extract information about the parameters \\( a \\), \\( b \\), \\( c \\), and \\( d \\).\n\nFirst, observe the amplitude \\( a \\). The amplitude is the maximum deviation from the midline of the sine wave. From the graph, the maximum value is 3 and the minimum value is -1. The midline is the average of these values:\n\\[\n\\text{Midline} = \\frac{3 + (-1)}{2} = 1\n\\]\nThe amplitude \\( a \\) is the distance from the midline to the maximum value:\n\\[\na = 3 - 1 = 2\n\\]\n\nNext, determine the period of the sine wave. The period \\( T \\) is the length of one complete cycle of the sine wave. From the graph, the period appears to be \\( \\frac{2\\pi}{3} \\). The period of the sine function \\( y = a \\sin (bx + c) + d \\) is given by:\n\\[\nT = \\frac{2\\pi}{b}\n\\]\nSetting this equal to the observed period:\n\\[\n\\frac{2\\pi}{b} = \\frac{2\\pi}{3}\n\\]\nSolving for \\( b \\):\n\\[\nb = 3\n\\]\n\nNow, determine the vertical shift \\( d \\). The vertical shift is the value of the midline:\n\\[\nd = 1\n\\]\n\nFinally, determine the phase shift \\( c \\). The phase shift is the horizontal shift of the sine wave. The standard sine function \\( y = \\sin x \\) starts at \\( x = 0 \\). For the function \\( y = 2 \\sin (3x + c) + 1 \\), the phase shift is given by:\n\\[\n\\text{Phase shift} = -\\frac{c}{b} = -\\frac{c}{3}\n\\]\nFrom the graph, the sine wave starts at \\( x = -\\frac{\\pi}{6} \\). Therefore, the phase shift is:\n\\[\n-\\frac{c}{3} = -\\frac{\\pi}{6}\n\\]\nSolving for \\( c \\):\n\\[\nc = \\frac{\\pi}{2}\n\\]\n\nThus, the smallest possible value of \\( c \\) is:\n\\[\n\\boxed{\\frac{\\pi}{2}}\n\\]\n\n**Output:** To determine the smallest possible value of \\( c \\) for the function \\( y = a \\sin (bx + c) + d \\), we analyze the graph to extract \\( a \\), \\( b \\), \\( c \\), and \\( d \\).\n\nFirst, observe the amplitude \\( a \\). The maximum value is 3 and the minimum is -1, so the midline is:\n\\[\n\text{Midline} = \x0crac{3 + (-1)}{2} = 1\n\\]\n\\[\na = 3 - 1 = 2\n\\]\n\nNext, determine the period. From the graph, the period is \\( \x0crac{2\\pi}{3} \\). Since\n\\[\n\x0crac{2\\pi}{b} = \x0crac{2\\pi}{3} \\Rightarrow b = 3\n\\]\n\nNow, determine the vertical shift:\n\\[\nd = 1\n\\]\n\nFinally, determine the phase shift. The function is \\( y = 2 \\sin (3x + c) + 1 \\), so the phase shift is:\n\\[\n-\x0crac{c}{3}\n\\]\nFrom the graph, the sine wave starts at \\( x = -\x0crac{\\pi}{6} \\), so:\n\\[\n-\x0crac{c}{3} = -\x0crac{\\pi}{6} \\Rightarrow c = \x0crac{\\pi}{2}\n\\]\n\n\\[\n\x08oxed{\x0crac{\\pi}{2}}\n\\]\n---\n\n"""


input_jsonl = "/data/cuiluyi/openr/results/MATH/best_of_n/20250404_154236/record.jsonl"
# input_jsonl = "/data/cuiluyi/openr/results/MATH/best_of_n/20250404_154236/demo.jsonl"
output_jsonl = "/data/cuiluyi/openr/results/MATH/best_of_n/20250404_154236/streamline.jsonl"

with jsonlines.open(input_jsonl) as reader, jsonlines.open(output_jsonl, mode='w') as writer:
    for idx, item in tqdm(enumerate(reader), total=12405):
        # 提取当前任务内容
        current_task = item["output"][0]["text"]
        
        # 动态构造完整提示
        full_prompt = (
            prompt_prefix +
            f"**Actual Task Input:** {current_task}\n"
            "---\n\n"
            "**Now, output the simplified solution for that exact input.**"
        )
        
        try:
            # 调用API
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 建议使用最新可用模型
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=2048
            )
            
            # 处理响应
            simplified_solution = response.choices[0].message.content
            
            # 写入结果
            writer.write({
                "original_task": current_task,
                "simplified_solution": simplified_solution,
            })
            
        except Exception as e:
            print(f"处理条目 {idx} 时发生错误：{str(e)}")
            # 记录错误条目
            writer.write({
                "original_task": current_task,
                "error": str(e),
                "timestamp": time.time()
            })