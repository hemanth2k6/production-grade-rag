import os
import requests
from dotenv import load_dotenv
import time
import sys

load_dotenv()

RENDER_API_KEY = os.getenv("RENDER_API_KEY")
if not RENDER_API_KEY:
    print("RENDER_API_KEY not found in .env")
    sys.exit(1)

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {RENDER_API_KEY}"
}

print("Fetching Render ownerId...")
response = requests.get("https://api.render.com/v1/owners", headers=HEADERS)
if response.status_code != 200:
    print(f"Failed to fetch owner info: {response.text}")
    sys.exit(1)

owner_id = response.json()[0]["owner"]["id"]

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
        "env": "python",
        "envSpecificDetails": {
            "buildCommand": "pip install -r requirements.txt",
            "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
        },
        "envVars": [
            {"key": "GEMINI_API_KEY", "value": os.getenv("GEMINI_API_KEY", "")},
            {"key": "LANGFUSE_PUBLIC_KEY", "value": os.getenv("LANGFUSE_PUBLIC_KEY", "")},
            {"key": "LANGFUSE_SECRET_KEY", "value": os.getenv("LANGFUSE_SECRET_KEY", "")},
            {"key": "LANGFUSE_HOST", "value": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")}
        ]
    }
}

response = requests.post("https://api.render.com/v1/services", json=payload, headers=HEADERS)
if response.status_code not in (200, 201):
    print(f"Failed to create service: {response.text}")
    sys.exit(1)

service_data = response.json()
service_id = service_data["id"]
service_url = service_data.get("service", {}).get("url", f"https://{payload['name']}.onrender.com")

print(f"Service created successfully! ID: {service_id}")
print(f"URL: {service_url}")

# Poll the deployment status
print("Polling deploy status...")
timeout = 600  # 10 minutes max
start_time = time.time()

# Get the initial deploy ID
deploy_id = None
time.sleep(5)
while time.time() - start_time < timeout:
    deploys_response = requests.get(f"https://api.render.com/v1/services/{service_id}/deploys", headers=HEADERS)
    if deploys_response.status_code == 200:
        deploys = deploys_response.json()
        if deploys:
            deploy_info = deploys[0]["deploy"]
            status = deploy_info["status"]
            print(f"[{int(time.time() - start_time)}s] Deploy status: {status}")
            if status == "live":
                print(f"Deployment LIVE at {service_url}")
                sys.exit(0)
            elif status in ["build_failed", "update_failed", "canceled", "deactivated"]:
                print(f"Deployment failed with status: {status}")
                sys.exit(1)
    else:
        print(f"Error fetching deploys: {deploys_response.text}")
        
    time.sleep(15)

print("Timeout waiting for deployment.")
sys.exit(1)
