import os
import requests
from dotenv import load_dotenv
import time

load_dotenv()

RENDER_API_KEY = os.getenv("RENDER_API_KEY")
if not RENDER_API_KEY:
    print("RENDER_API_KEY not found in .env")
    exit(1)

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {RENDER_API_KEY}"
}

# 1. Get Owner ID (we need this to create a service)
print("Fetching Render ownerId...")
response = requests.get("https://api.render.com/v1/owners", headers=HEADERS)
if response.status_code != 200:
    print(f"Failed to fetch owner info: {response.text}")
    exit(1)

owner_id = response.json()[0]["owner"]["id"]

# 2. Create the Web Service
print(f"Creating Web Service for owner {owner_id}...")

payload = {
    "type": "web_service",
    "name": "production-grade-rag-service",
    "ownerId": owner_id,
    "repo": "https://github.com/hemanth2k6/production-grade-rag",
    "branch": "main",
    "env": "python",
    "plan": "free",
    "region": "oregon",
    "serviceDetails": {
        "buildCommand": "pip install -r requirements.txt",
        "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
        "envVars": [
            {"key": "SUPABASE_URL", "value": os.getenv("SUPABASE_URL", "")},
            {"key": "SUPABASE_KEY", "value": os.getenv("SUPABASE_KEY", "")},
            {"key": "OPENROUTER_API_KEY", "value": os.getenv("OPENROUTER_API_KEY", "")},
            {"key": "LANGFUSE_PUBLIC_KEY", "value": os.getenv("LANGFUSE_PUBLIC_KEY", "")},
            {"key": "LANGFUSE_SECRET_KEY", "value": os.getenv("LANGFUSE_SECRET_KEY", "")},
            {"key": "LANGFUSE_HOST", "value": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")}
        ]
    }
}

response = requests.post("https://api.render.com/v1/services", json=payload, headers=HEADERS)
if response.status_code not in (200, 201):
    print(f"Failed to create service: {response.text}")
    exit(1)

service_data = response.json()
service_id = service_data["id"]
service_url = service_data["service"]["url"]
print(f"Service created successfully! ID: {service_id}")
print(f"Live URL will be: {service_url}")

# Note: Render automatically starts the first deploy when creating a service linked to a repo.
# We will just print the URL. The deployment takes a few minutes.
print("Triggering initial deployment if not already started...")
requests.post(f"https://api.render.com/v1/services/{service_id}/deploys", headers=HEADERS)

print("Deployment initiated. You can check the dashboard at dashboard.render.com")
