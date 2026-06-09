#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "⚙️  Initializing Smart City Production Deployment..."

# 1. Ensure the system environment hidden config folder exists
echo "📂 Re-verifying local runtime directory architectures..."
mkdir -p .streamlit/

# 2. Re-verify configuration variables (fallback check)
if [ ! -f .streamlit/config.toml ]; then
    echo "📝 Configuration file not found. Generating default headless configurations..."
    cat << EOF > .streamlit/config.toml
[server]
headless = true
enableCORS = false
port = $PORT
EOF
fi

# 3. Complete system package installation
echo "📦 Installing production dependencies listed in requirements.txt..."
pip install --no-cache-dir -r requirements.txt

echo "🚀 Launching Multi-Agent Streamlit Web Server Application Core..."
# 4. Fire up the application on the dynamically assigned port from the cloud provider
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
