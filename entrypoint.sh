#!/bin/bash
set -e

# Check if processed data exists
if [ ! -f "processed_movies.pkl" ] || [ ! -d "indices" ]; then
    echo "Processed data not found. Running data processor..."
    python src/data_processor.py
else
    echo "Processed data found. Skipping processing."
fi

# Run the main application
echo "Starting Chatbot..."
exec python main.py
