#!/bin/bash
cd /home/site/wwwroot
pip install -r requirements.txt
cd app
uvicorn main:app --host 0.0.0.0 --port 8000