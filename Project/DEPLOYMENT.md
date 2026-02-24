# Sentira AI - Deployment Instructions

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python app.py
   ```

## Production Deployment

### Render (Automated, Recommended)

This project is configured for Render Blueprint deployment with:
- `render.yaml` at repository root
- `Project/Dockerfile`
- Health endpoint: `GET /health`

Steps:
1. Push repository to GitHub.
2. In Render dashboard, select `New` -> `Blueprint`.
3. Connect your repo and deploy.
4. Future pushes auto-deploy (enabled in `render.yaml`).

### Using Gunicorn (Recommended)
1. Install Gunicorn:
   ```bash
   pip install gunicorn
   ```
2. Run with Gunicorn:
   ```bash
   gunicorn app:app --bind 0.0.0.0:5000 --timeout 120
   ```

### Deploy to Heroku
1. Install Heroku CLI and login.
2. In the `Project` directory:
   ```bash
   heroku create
   git add .
   git commit -m "Deploy Sentira AI"
   git push heroku main
   ```

### Deploy with Docker
1. Build the Docker image:
   ```bash
   docker build -t sentira-ai .
   ```
2. Run the container:
   ```bash
   docker run -p 5000:5000 sentira-ai
   ```

### Google App Engine (Flexible)
1. Install Google Cloud SDK and initialize.
2. Deploy:
   ```bash
   gcloud app deploy app.yaml
   ```

---

- Ensure your model files (`.keras`) are present in the `Project` directory.
- For static files, Flask will serve from `/static` and `/templates`.
- For custom domains or HTTPS, refer to your cloud provider's documentation.
