# Ensure your API key is set as an environment variable
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4o-mini-2024-07-18",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "what is your name!"}
  ]
)

print(completion.choices[0].message)