import requests

API_KEY = "0cfe7441cc466c7c201a0afa04047da7"  # Replace with your real API key
url = "https://chat-ai.academiccloud.de/v1/embeddings"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "input": "Hello world",
    "model": "e5-mistral-7b-instruct",
    "encoding_format": "float"
}

response = requests.post(url, headers=headers, json=payload)

print("Status code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Failed to parse JSON:", e)
