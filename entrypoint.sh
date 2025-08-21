#!/bin/bash

# Ensure HF token is set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
  echo "Error: HUGGINGFACE_TOKEN environment variable not set."
  exit 1
fi

# Get prompt from input
PROMPT="$*"
if [ -z "$PROMPT" ]; then
  echo "Error: No prompt provided."
  exit 1
fi

# Run the Python script
python3 run_llm.py "$PROMPT"
