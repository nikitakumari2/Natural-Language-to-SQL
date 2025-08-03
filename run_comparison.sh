#!/bin/bash

# Script to run SQL Copilot model comparisons

# Example usage:
# ./run_comparison.sh "How many students have a GPA above 3.5?"

if [ -z "$1" ]; then
  echo "Please provide a question to ask the SQL Copilot"
  echo "Usage: ./run_comparison.sh \"Your question here\""
  exit 1
fi

# Run the comparison with all available models
python enhanced_sql_copilot.py --question "$1" --compare

# Alternatively, specify specific models to compare
# python enhanced_sql_copilot.py --question "$1" --compare --models "gpt-3.5-turbo,gpt-4,mixtral-8x7b"