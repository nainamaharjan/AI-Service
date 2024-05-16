#!/usr/bin/env bash
export PYTHONPATH=`pwd`
UI_PORT=${UI_PORT:-8088}
streamlit run ui/Homepage.py --server.port ${UI_PORT}
