def merge_dict_list(dict_list):
    if not dict_list:
        return {}

    merged_dict = {
        "error_code": dict_list[0]["error_code"],
        "text": [],
        "usage": {
            "prompt_tokens": dict_list[0]["usage"]["prompt_tokens"],
            "completion_tokens": sum(
                d["usage"]["completion_tokens"] for d in dict_list
            ),
            "total_tokens": sum(d["usage"]["total_tokens"] for d in dict_list),
        },
        "cumulative_logprob": [],
        "output_token_len": [],
        "finish_reason": [],
    }

    # 合并 text 字段
    merged_dict["text"] = [item for d in dict_list for item in d["text"]]

    # 合并 cumulative_logprob 字段
    merged_dict["cumulative_logprob"] = [
        logprob for d in dict_list for logprob in d["cumulative_logprob"]
    ]

    # 合并 output_token_len 字段
    merged_dict["output_token_len"] = [
        otl for d in dict_list for otl in d["output_token_len"]
    ]

    # 合并 finish_reason 字段（处理列表或单个值）
    merged_dict["finish_reason"] = [
        reason
        for d in dict_list
        for reason in (
            d["finish_reason"]
            if isinstance(d["finish_reason"], list)
            else [d["finish_reason"]]
        )
    ]

    return merged_dict


if __name__ == "__main__":
    data = [
        {
            "text": [
                "To convert the point \\((0,3)\\) from rectangular coordinates to polar coordinates, we need to find the values of \\(r\\) and \\(\\theta\\). The formulas for converting from rectangular coordinates \\((x, y)\\) to polar coordinates \\((r, \\theta)\\) are:\n\n\\[\nr = \\sqrt{x^2 + y^2}\n\\]\n\\[\n\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\n\\]\n\nGiven the point \\((0,3)\\), we have \\(x = 0\\) and \\(y = 3\\).\n\nFirst, we calculate \\(r\\):\n\n\\[\nr = \\sqrt{0^2 + 3^2} = \\sqrt{0 + 9} = \\sqrt{9} = 3\n\\]\n\nNext, we calculate \\(\\theta\\). The formula \\(\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\\) becomes \\(\\theta = \\tan^{-1}\\left(\\frac{3}{0}\\right)\\). Since division by zero is undefined, we need to consider the position of the point \\((0,3)\\) in the coordinate system. The point \\((0,3)\\) lies on the positive \\(y\\)-axis. Therefore, the angle \\(\\theta\\) is \\(\\frac{\\pi}{2}\\) radians.\n\nThus, the polar coordinates of the point \\((0,3)\\) are:\n\n\\[\n(r, \\theta) = \\left(3, \\frac{\\pi}{2}\\right)\n\\]\n\nSo, the final answer is:\n\n\\[\n\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}\n\\]"
            ],
            "error_code": 0,
            "usage": {
                "prompt_tokens": 77,
                "completion_tokens": 358,
                "total_tokens": 435,
            },
            "cumulative_logprob": [0.0],
            "output_token_len": [358],
            "finish_reason": "stop",
        },
        {
            "text": [
                "To convert the point \\((0,3)\\) from rectangular coordinates to polar coordinates, we need to find the values of \\(r\\) and \\(\\theta\\). The formulas for converting from rectangular coordinates \\((x, y)\\) to polar coordinates \\((r, \\theta)\\) are:\n\n\\[\nr = \\sqrt{x^2 + y^2}\n\\]\n\\[\n\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\n\\]\n\nGiven the point \\((0, 3)\\), we have \\(x = 0\\) and \\(y = 3\\).\n\nFirst, we calculate \\(r\\):\n\n\\[\nr = \\sqrt{0^2 + 3^2} = \\sqrt{0 + 9} = \\sqrt{9} = 3\n\\]\n\nNext, we calculate \\(\\theta\\). The formula \\(\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\\) becomes:\n\n\\[\n\\theta = \\tan^{-1}\\left(\\frac{3}{0}\\right)\n\\]\n\nSince division by zero is undefined, we need to consider the position of the point \\((0, 3)\\) in the coordinate plane. The point \\((0, 3)\\) lies on the positive \\(y\\)-axis. The angle \\(\\theta\\) for a point on the positive \\(y\\)-axis is \\(\\frac{\\pi}{2}\\).\n\nThus, \\(\\theta = \\frac{\\pi}{2}\\).\n\nTherefore, the polar coordinates of the point \\((0, 3)\\) are:\n\n\\[\n(r, \\theta) = \\left(3, \\frac{\\pi}{2}\\right)\n\\]\n\nSo, the final answer is:\n\n\\[\n\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}\n\\]"
            ],
            "error_code": 0,
            "usage": {
                "prompt_tokens": 77,
                "completion_tokens": 386,
                "total_tokens": 463,
            },
            "cumulative_logprob": [0.0],
            "output_token_len": [386],
            "finish_reason": "stop",
        },
        {
            "text": [
                "To convert the point \\((0,3)\\) from rectangular coordinates to polar coordinates, we need to find the values of \\(r\\) and \\(\\theta\\). The formulas for converting from rectangular coordinates \\((x, y)\\) to polar coordinates \\((r, \\theta)\\) are:\n\n\\[\nr = \\sqrt{x^2 + y^2}\n\\]\n\\[\n\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\n\\]\n\nGiven the point \\((0,3)\\), we have \\(x = 0\\) and \\(y = 3\\).\n\nFirst, we calculate \\(r\\):\n\n\\[\nr = \\sqrt{0^2 + 3^2} = \\sqrt{0 + 9} = \\sqrt{9} = 3\n\\]\n\nNext, we calculate \\(\\theta\\). The formula \\(\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\\) becomes \\(\\theta = \\tan^{-1}\\left(\\frac{3}{0}\\right)\\). Since division by zero is undefined, we need to consider the position of the point \\((0,3)\\) in the coordinate system. The point \\((0,3)\\) lies on the positive \\(y\\)-axis. Therefore, the angle \\(\\theta\\) is \\(\\frac{\\pi}{2}\\) radians.\n\nThus, the polar coordinates of the point \\((0,3)\\) are:\n\n\\[\n(r, \\theta) = \\left(3, \\frac{\\pi}{2}\\right)\n\\]\n\nSo, thefinal answer is:\n\n\\[\n\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}\n\\]",
                "To convert the point \\((0,3)\\) from rectangular coordinates to polar coordinates, we need to find the values of \\(r\\) and \\(\\theta\\). The formulas for converting from rectangular coordinates \\((x, y)\\) to polar coordinates \\((r, \\theta)\\) are:\n\n\\[\nr = \\sqrt{x^2 + y^2}\n\\]\n\\[\n\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\n\\]\n\nGiven the point \\((0, 3)\\), we have \\(x = 0\\) and \\(y = 3\\).\n\nFirst, we calculate \\(r\\):\n\n\\[\nr = \\sqrt{0^2 + 3^2} = \\sqrt{0 + 9} = \\sqrt{9} = 3\n\\]\n\nNext, we calculate \\(\\theta\\). The formula \\(\\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\\) becomes:\n\n\\[\n\\theta = \\tan^{-1}\\left(\\frac{3}{0}\\right)\n\\]\n\nSince division by zero is undefined, we need to consider the position of the point \\((0, 3)\\) in the coordinate plane. The point \\((0, 3)\\) lies on the positive \\(y\\)-axis. The angle \\(\\theta\\) for a point on the positive \\(y\\)-axis is \\(\\frac{\\pi}{2}\\).\n\nThus, \\(\\theta = \\frac{\\pi}{2}\\).\n\nTherefore, the polar coordinates of the point \\((0, 3)\\) are:\n\n\\[\n(r, \\theta) = \\left(3, \\frac{\\pi}{2}\\right)\n\\]\n\nSo, the final answer is:\n\n\\[\n\\boxed{\\left(3, \\frac{\\pi}{2}\\right)}\n\\]",
            ],
            "error_code": 0,
            "usage": {
                "prompt_tokens": 77,
                "completion_tokens": 744,
                "total_tokens": 898,
            },
            "cumulative_logprob": [0.0, 0.0],
            "output_token_len": [358, 386],
            "finish_reason": ["stop", "stop"],
        },
    ]

    print(merge_dict_list([] + data))
