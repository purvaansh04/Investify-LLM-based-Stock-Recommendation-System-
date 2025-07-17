#!/bin/bash

# Export environment variables
export OLLAMA_HOST="0.0.0.0:11500"
export OLLAMA_MODELS="$HOME/.ollama/models"

# Optional: print the environment variables for confirmation
echo "Starting Ollama server with:"
echo "OLLAMA_HOST=$OLLAMA_HOST"
echo "OLLAMA_MODELS=$OLLAMA_MODELS"

# Start the Ollama server
ollama serve
