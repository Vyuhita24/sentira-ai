# Sentira AI

**Tagline:** Real-Time Voice Emotion & Stress Intelligence

Sentira AI is a Flask-based web application that analyzes voice audio to infer emotional state and stress level using a trained deep learning model.

## Features

- Live microphone capture and analysis
- WAV file upload and model inference
- Emotion prediction with confidence score
- Stress-state interpretation for each prediction

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python app.py
   ```
3. Open:
   ```text
   http://127.0.0.1:5000
   ```

## API Endpoint

- `POST /predict_audio`
  - Form field: `file` (WAV audio)
  - Returns JSON with `result`, `confidence`, and `emotion`

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for production deployment options.

## Render (Automated Docker Deploy)

This repository is preconfigured for Render Blueprint deployment via [`render.yaml`](../render.yaml).

1. Push this project to GitHub.
2. In Render, choose `New` -> `Blueprint`.
3. Connect the GitHub repo and select this repository.
4. Render will detect `render.yaml`, build the Docker image from `Project/Dockerfile`, and deploy.
5. Every push to the tracked branch auto-deploys (`autoDeploy: true`).

Service health is exposed at:
- `GET /health`
