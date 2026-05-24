#!/bin/bash
echo "🚀 Starting EchoLang..."
cd backend
pip install -r requirements.txt -q
echo "✅ Dependencies installed"
python download_models.py
echo "✅ Models downloaded"
# Run server in background
uvicorn main:app --host [IP_ADDRESS] --port 8000 --reload &
