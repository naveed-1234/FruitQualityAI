# Fruit Quality AI — Render deployment

## Render settings
- Service type: Web Service
- Runtime: Python 3
- Build command: `pip install --upgrade pip && pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 300 app:app`
- Plan: Free
- Python: 3.11.11

The repository includes `render.yaml` with these settings. If creating the service from the Render Blueprint flow, Render can read this file. Otherwise enter the settings above manually.

## Environment variables
- `GEMINI_API_KEY`: optional. Add this in Render's Environment tab if the chatbot should use Gemini. Never commit the key to GitHub.
- `TF_CPP_MIN_LOG_LEVEL=2`

## Important free-plan limitations
Render free web services have 512 MB RAM and spin down after 15 minutes without inbound traffic. This project loads two TensorFlow/Keras models when the Flask app starts, so it may exceed the free instance's memory. If it restarts or fails with an out-of-memory error, the current models/features may not fit Render Free. Do not assume deployment succeeded until the logs show `Models loaded successfully!` and image prediction works.

The local `static/uploads/` directory is ephemeral on Render Free. Uploaded images and Grad-CAM output may be deleted on restart/redeploy; do not use it for permanent user data.

## Included files
The two `.keras` model files, Flask app, templates, static assets, Grad-CAM code, and shelf-life logic are included. Local uploads, virtual environments, secrets, and Python cache files are excluded.
