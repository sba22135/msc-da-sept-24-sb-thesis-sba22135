#!/usr/bin/env bash
set -e
# Activate venv if present
source venv/bin/activate 2>/dev/null || true
# Launch Streamlit dashboard
streamlit run app.py
