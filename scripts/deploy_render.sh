#!/bin/bash
source .env

if [ -z "$RENDER_API_KEY" ]; then
    echo "RENDER_API_KEY not found in .env"
    exit 1
fi

HEADERS=(
    -H "accept: application/json"
    -H "content-type: application/json"
    -H "authorization: Bearer $RENDER_API_KEY"
)

echo "Fetching ownerId..."
USER_JSON=$(curl -s -X GET "https://api.render.com/v1/users" "${HEADERS[@]}")
OWNER_ID=$(echo $USER_JSON | grep -o '"id":"[^"]*' | head -1 | cut -d'"' -f4)

if [ -z "$OWNER_ID" ]; then
    echo "Failed to fetch owner ID. Response: $USER_JSON"
    exit 1
fi

echo "Creating service for owner $OWNER_ID..."

PAYLOAD=$(cat <<EOF
{
    "type": "web_service",
    "name": "production-grade-rag",
    "ownerId": "$OWNER_ID",
    "repo": "https://github.com/hemanth2k6/production-grade-rag",
    "branch": "main",
    "env": "docker",
    "plan": "free",
    "region": "oregon",
    "serviceDetails": {
        "envVars": [
            {"key": "SUPABASE_URL", "value": "$SUPABASE_URL"},
            {"key": "SUPABASE_KEY", "value": "$SUPABASE_KEY"},
            {"key": "GEMINI_API_KEY", "value": "$GEMINI_API_KEY"},
            {"key": "LANGFUSE_PUBLIC_KEY", "value": "$LANGFUSE_PUBLIC_KEY"},
            {"key": "LANGFUSE_SECRET_KEY", "value": "$LANGFUSE_SECRET_KEY"},
            {"key": "LANGFUSE_HOST", "value": "$LANGFUSE_HOST"}
        ]
    }
}
EOF
)

SERVICE_JSON=$(curl -s -X POST "https://api.render.com/v1/services" "${HEADERS[@]}" -d "$PAYLOAD")
SERVICE_ID=$(echo $SERVICE_JSON | grep -o '"id":"[^"]*' | head -1 | cut -d'"' -f4)
SERVICE_URL=$(echo $SERVICE_JSON | grep -o '"url":"[^"]*' | cut -d'"' -f4)

if [ -z "$SERVICE_ID" ]; then
    echo "Failed to create service. Response: $SERVICE_JSON"
    exit 1
fi

echo "Service created! ID: $SERVICE_ID"
echo "Live URL: $SERVICE_URL"

echo "Triggering deploy..."
curl -s -X POST "https://api.render.com/v1/services/$SERVICE_ID/deploys" "${HEADERS[@]}"
echo "Done."
