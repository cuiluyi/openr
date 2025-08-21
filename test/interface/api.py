from openai import OpenAI
from constants import API_KEY, BASE_URL, MODEL_NAME

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

completion = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
)
print(completion.choices[0].message.content)