import re


SYSTEM1_BEGIN_TAG = "<｜begin▁of▁system1｜>"
SYSTEM2_BEGIN_TAG = "<｜begin▁of▁system2｜>"
SYSTEM1_END_TAG = "<｜end▁of▁system1｜>"
SYSTEM2_END_TAG = "<｜end▁of▁system2｜>"
SUFFIX = "<｜end▁of▁sentence｜>"


def has_valid_content(text: str) -> bool:
    for tag in [SYSTEM1_BEGIN_TAG, SYSTEM2_BEGIN_TAG, SYSTEM1_END_TAG, SYSTEM2_END_TAG, SUFFIX]:
        text = text.replace(tag, "").strip()
    return bool(text)


def split_string(input_string, delimiters):
    escaped_delimiters = [re.escape(d) for d in delimiters]
    pattern = "(" + "|".join(escaped_delimiters) + ")"
    parts = re.split(pattern, input_string, maxsplit=1)

    if len(parts) == 1:
        return parts[0], "", ""
    else:
        return parts[0], parts[1], parts[2]


if __name__ == "__main__":
    test_cases = [
        "hello SYSTEM1_BEGIN_TAG world SUFFIX end",
        "start SUFFIX middle SYSTEM2_BEGIN_TAG end",
        "no delimiters here",
    ]

    for test in test_cases:
        first_part, delimiter, remaining = split_string(
            test, ["SYSTEM1_BEGIN_TAG", "SYSTEM2_BEGIN_TAG", "SUFFIX"]
        )
        print(f"Input: {test}")
        print(f"First part: {repr(first_part)}")
        print(f"Delimiter: {repr(delimiter)}")
        print(f"Remaining: {repr(remaining)}")
