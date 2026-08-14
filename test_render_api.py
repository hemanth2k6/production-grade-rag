import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("RENDER_API_KEY")
owner_id = requests.get("https://api.render.com/v1/owners", headers={"Authorization": f"Bearer {api_key}"}).json()[0]["owner"]["id"]

payload = {
    "type": "web_service",
    "name": "test-rag-service-debug",
    "ownerId": owner_id,
    "repo": "https://github.com/hemanth2k6/production-grade-rag",
    "branch": "main",
    "env": "docker"
}
r = requests.post("https://api.render.com/v1/services", json=payload, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
print(r.status_code, r.text)
