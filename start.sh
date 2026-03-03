#!/bin/bash
set -e

# Start Discord bot in background
echo "🤖 Starting Discord bot..."
python bot/discord_bot.py &
BOT_PID=$!

# Start backend API
echo "🚀 Starting MoonWire backend..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for both processes
wait $BOT_PID $API_PID
