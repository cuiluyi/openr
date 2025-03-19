import requests

controller_addr = "http://0.0.0.0:28777"
model_name = "Qwen2.5-Math-PRM-7B"

input_str = [
    "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$&&&&&To convert the point \\((0,3)\\) from rectangular coordinates to polar coordinates, we need to find the values of \\(r\\) and \\(\\theta\\). The formulas for the conversion are: <extra_0>\\[ r = \\sqrt{x^2 + y^2} \\]\n\\[ \\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right) \\] <extra_0>Given the point \\((0,3)\\), we have \\(x = 0\\) and \\(y = 3\\). <extra_0>First, we calculate \\(r\\):\n\\[ r = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3 \\] <extra_0>Next, we calculate \\(\\theta\\). The formula \\(\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\\) is indeterminate for \\(x = 0\\), so we need to consider the position of the point in the coordinate plane. The point \\((0,3)\\) lies on the positive \\(y\\)-axis, so \\(\\theta = \\frac{\\pi}{2}\\). <extra_0>Therefore, the polar coordinates of the point \\((0,3)\\) are:\n\\[ (3, \\frac{\\pi}{2}) \\] <extra_0>So the final answer is:\n\\[ \\boxed{\\left(3, \\frac{\\pi}{2}\\right)} \\] <extra_0>",
    "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$&&&&&To convert the point \\((0,3)\\) from rectangular coordinates to polar coordinates, we need to find the values of \\(r\\) and \\(\\theta\\). The formulas for the conversion are: <extra_0>\\[ r = \\sqrt{x^2 + y^2} \\]\n\\[ \\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right) \\] <extra_0>For the point \\((0,3)\\), the coordinates are \\(x = 0\\) and \\(y = 3\\). Let's calculate \\(r\\) first: <extra_0>\\[ r = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3 \\] <extra_0>Next, we calculate \\(\\theta\\). The formula \\(\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\\) becomes \\(\\theta = \\tan^{-1}\\left(\\frac{3}{0}\\right)\\). Since division by zero is undefined, we need to consider the location of the point. The point \\((0,3)\\) is on the positive \\(y\\)-axis, so \\(\\theta\\) is \\(\\frac{\\pi}{2}\\). <extra_0>Therefore, the polar coordinates are: <extra_0>\\[ (r, \\theta) = \\left(3, \\frac{\\pi}{2}\\right) \\] <extra_0>The final answer is: <extra_0>\\[ \\boxed{\\left(3, \\frac{\\pi}{2}\\right)} \\] <extra_0>",
    "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$&&&&&To convert the point \\((0,3)\\) from rectangular coordinates to polar coordinates, we need to find the values of \\(r\\) and \\(\\theta\\). The formulas for converting from rectangular coordinates \\((x, y)\\) to polar coordinates \\((r, \\theta)\\) are: <extra_0>\\[ r = \\sqrt{x^2 + y^2} \\]\n\\[ \\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right) \\] <extra_0>For the point \\((0,3)\\), the coordinates are \\(x = 0\\) and \\(y = 3\\). <extra_0>First, we calculate \\(r\\): <extra_0>\\[ r = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3 \\] <extra_0>Next, we calculate \\(\\theta\\). The formula \\(\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\\) would suggest \\(\\theta = \\tan^{-1}\\left(\\frac{3}{0}\\right)\\), which is undefined because division by zero is not allowed. However, we know that the point \\((0,3)\\) lies on the positive \\(y\\)-axis. Therefore, the angle \\(\\theta\\) is \\(\\frac{\\pi}{2}\\) radians. <extra_0>So, the polar coordinates of the point \\((0,3)\\) are: <extra_0>\\[ (3, \\frac{\\pi}{2}) \\] <extra_0>Thus, the answer is: <extra_0>\\[\n\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}\n\\] <extra_0>",
]


ret = requests.post(controller_addr + "/get_worker_address", json={"model": model_name})
worker_addr = ret.json()["address"]
if not worker_addr:
    raise ValueError("Value Model name {} does not exist.".format(model_name))

headers = {"User-Agent": "FastChat Client"}
gen_params = {"input_str": input_str}
response = requests.post(
    worker_addr + "/worker_value_inference",
    headers=headers,
    json=gen_params,
    stream=True,
)

print(response.json())
