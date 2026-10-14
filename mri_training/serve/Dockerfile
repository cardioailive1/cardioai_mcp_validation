# CardioAI cardiac MRI model serving container (Vertex AI custom container).
# Build from the REPOSITORY ROOT so the runtime/ code is copied unchanged:
#   docker build -f mri_training/serve/Dockerfile -t <image> .
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir "numpy<3" "scipy>=1.10" "pydicom>=2.4" flask google-cloud-storage \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Same files as training; the predictor refuses to start if their fingerprint differs from the model card.
COPY mri_training/__init__.py mri_training/__init__.py
COPY mri_training/io.py mri_training/io.py
COPY mri_training/runtime/ mri_training/runtime/
COPY mri_training/serve/predictor.py mri_training/serve/predictor.py

ENV PYTHONUNBUFFERED=1 AIP_HTTP_PORT=8080 AIP_HEALTH_ROUTE=/health AIP_PREDICT_ROUTE=/predict
EXPOSE 8080
RUN useradd -m -u 1000 predictor
USER predictor
CMD ["python", "mri_training/serve/predictor.py"]
