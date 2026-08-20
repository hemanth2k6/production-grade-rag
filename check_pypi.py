import urllib.request
import json

for pkg in ["google-generativeai", "langchain-google-genai", "google-genai"]:
    url = f"https://pypi.org/pypi/{pkg}/json"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(f"{pkg} latest version: {data['info']['version']}")
