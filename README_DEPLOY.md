# Fruit Quality AI — Cloud Run deployment

This package keeps the existing Flask UI, two Keras models, Grad-CAM, shelf-life rules, and Gemini chatbot. It excludes the local virtual environment, Python caches, and the contents of `static/uploads/` (which included local test images).

## Before deploying

1. Install Google Cloud CLI and Docker Desktop (Docker is useful for local testing).
2. Create/select a Google Cloud project and link a billing account. Cloud Run has a monthly free tier, but billing must be enabled and usage beyond the free tier can be charged. Set a budget alert before deployment.
3. Enable the Cloud Run and Cloud Build APIs.

## Local test (optional)

```bash
docker build -t fruit-quality-ai .
docker run --rm -p 8080:8080 -e GEMINI_API_KEY=YOUR_KEY fruit-quality-ai
```

Open http://localhost:8080. Do not put a real API key in source files or commit it.

## Deploy to Cloud Run

Run these commands in this project directory after `gcloud auth login` and `gcloud config set project YOUR_PROJECT_ID`:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy fruit-quality-ai \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 1 \
  --concurrency 1 \
  --max-instances 1 \
  --min 0 \
  --timeout 300 \
  --set-env-vars TF_CPP_MIN_LOG_LEVEL=2 \
  --set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest
```

Before the deploy command, create a Secret Manager secret named `GEMINI_API_KEY` and grant the Cloud Run service identity permission to access it. If you do not want to configure Gemini yet, omit the `--set-secrets` flag; the app has a local fallback chatbot response.

## Cost and storage notes

- Keep `--min 0` and `--max-instances 1`; do not configure minimum instances.
- The free Cloud Run request-based tier has monthly CPU, memory, and request allowances; it is not an unlimited hosting guarantee. TensorFlow startup and image inference consume resources.
- Google Cloud billing must be enabled. Set a billing budget alert and monitor actual usage. A budget alert is not a hard spending cap.
- Files in `static/uploads/` are temporary on Cloud Run and can disappear when an instance is replaced. The current app uses local file paths for uploads and Grad-CAM output; this is acceptable for a demo but not durable storage.
- Uploaded videos and large images are constrained by Cloud Run's HTTP request limits. Test with ordinary images first.

## Notes

The Keras models were saved with Keras 3.10.0, so this package pins Keras 3.10.0 and TensorFlow CPU 2.19.0. Verify both models load successfully in the build/runtime logs before sharing the URL.
