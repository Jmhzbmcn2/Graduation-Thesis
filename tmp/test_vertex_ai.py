import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from google import genai

client = genai.Client(
    vertexai=True,
    project="project-2f6e301d-eae6-49b1-94a",
    location="us-central1"
)

r = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Xin chao, tra loi ngan gon bang tieng Viet"
)

print("Response:", r.text)
print("Input tokens:", r.usage_metadata.prompt_token_count)
print("Output tokens:", r.usage_metadata.candidates_token_count)
