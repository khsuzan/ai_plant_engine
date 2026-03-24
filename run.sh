#!/bin/bash

# Simple wrapper to execute commands exactly like 'npm run'
COMMAND=$1
shift

if [ "$COMMAND" = "start" ]; then
    echo "Starting Local Server..."
    uv run uvicorn dummy_project.asgi:application --reload "$@"
elif [ "$COMMAND" = "harvest" ]; then
    echo "Running Harvester..."
    if [ $# -eq 0 ]; then
        uv run python manage.py run_harvester --count 5
    else
        uv run python manage.py run_harvester "$@"
    fi
elif [ "$COMMAND" = "test-engine" ]; then
    echo "Testing Harvester Engine Standalone..."
    uv run python harvester/engine.py "$@"
else
    echo "Usage: ./run.sh [start | harvest | test-engine] [args...]"
    echo ""
    echo "Available commands:"
    echo "  start       - Starts the Uvicorn ASGI server"
    echo "  harvest     - Runs the Django management command (default targets 5 plants)"
    echo "  test-engine - Runs the Python engine.py script directly bypassing Django"
fi
