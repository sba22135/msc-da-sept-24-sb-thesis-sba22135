#!/usr/bin/env bash
set -e
source venv/bin/activate 2>/dev/null || true
streamlit run app.py
