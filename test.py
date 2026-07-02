import requests

r = requests.post(
    "http://localhost:11435/v1/chat/completions",
    json={
        "model": "qwen3.5:latest",
        "messages": [
            {"role": "user", "content": "hello"}
        ]
    }
)

print(r.status_code)
print(r.text)