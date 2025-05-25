#!/bin/bash

# Set ChromaDB environment variables
export CHROMA_SERVER_AUTHN_PROVIDER=""
export CHROMA_SERVER_AUTHN_CREDENTIALS=""
export ANONYMIZED_TELEMETRY="False"
export CHROMA_SERVER_HOST="localhost"
export CHROMA_SERVER_HTTP_PORT="8000"
export IS_PERSISTENT="TRUE"

# Create necessary directories
mkdir -p /tmp/chroma_db
mkdir -p /app/.chroma

# Set permissions
chmod 755 /tmp/chroma_db
chmod 755 /app/.chroma

# Run the Streamlit app
streamlit run src/resume/main.py
