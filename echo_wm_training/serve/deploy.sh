#!/usr/bin/env bash
# Deploy an exported echo wall-motion bundle (weights.pt, thresholds.json, model_card.json) to Vertex AI.
# Usage: BUNDLE_URI=gs://<bucket>/echo-wm/v1 STUDY_PREFIX=gs://<bucket>/echo-studies/ IMAGE_TAG=v1 echo_wm_training/serve/deploy.sh
set -euo pipefail
PROJECT_ID="${PROJECT_ID:-ai-powered-cardiology}"
REGION="${REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-cardioai-echo-wm}"
IMAGE_TAG="${IMAGE_TAG:?set IMAGE_TAG}"
BUNDLE_URI="${BUNDLE_URI:?set BUNDLE_URI (gs:// folder with weights.pt, thresholds.json, model_card.json)}"
# Echo cines are passed by reference (Vertex online requests are limited to 1.5 MB); the endpoint's
# service account needs read access to this prefix and the predictor refuses any other location.
STUDY_PREFIX="${STUDY_PREFIX:?set STUDY_PREFIX (gs:// folder the app uploads echo cines or study zips to)}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"   # up to 9 clips of 32 frames; add a GPU for < 5 s per study
MODEL_NAME="${MODEL_NAME:-cardioai-echo-wm-mobilenetv3}"
ENDPOINT_NAME="${ENDPOINT_NAME:-cardioai-echo-wm-mobilenetv3}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/echo-wm-mobilenetv3:${IMAGE_TAG}"

for f in weights.pt thresholds.json model_card.json; do
  gsutil -q stat "${BUNDLE_URI%/}/${f}" || { echo "missing ${f} in ${BUNDLE_URI}"; exit 1; }
done

cd "$(git rev-parse --show-toplevel)"
docker build -f echo_wm_training/serve/Dockerfile -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"

gcloud ai models upload --project="${PROJECT_ID}" --region="${REGION}" \
  --display-name="${MODEL_NAME}" --container-image-uri="${IMAGE_URI}" \
  --container-health-route=/health --container-predict-route=/predict --container-ports=8080 \
  --container-env-vars="ALLOWED_STUDY_PREFIXES=${STUDY_PREFIX}" \
  --artifact-uri="${BUNDLE_URI}"

MODEL_ID="$(gcloud ai models list --project="${PROJECT_ID}" --region="${REGION}" \
  --filter="displayName=${MODEL_NAME}" --sort-by=~createTime --format='value(name.basename())' | head -1)"
ENDPOINT_ID="$(gcloud ai endpoints list --project="${PROJECT_ID}" --region="${REGION}" \
  --filter="displayName=${ENDPOINT_NAME}" --format='value(name.basename())' | head -1)"
if [[ -z "${ENDPOINT_ID}" ]]; then
  gcloud ai endpoints create --project="${PROJECT_ID}" --region="${REGION}" --display-name="${ENDPOINT_NAME}"
  ENDPOINT_ID="$(gcloud ai endpoints list --project="${PROJECT_ID}" --region="${REGION}" \
    --filter="displayName=${ENDPOINT_NAME}" --format='value(name.basename())' | head -1)"
fi

# One model per endpoint, 100% traffic (the app's lock check rejects shared endpoints).
gcloud ai endpoints deploy-model "${ENDPOINT_ID}" --project="${PROJECT_ID}" --region="${REGION}" \
  --model="${MODEL_ID}" --display-name="${MODEL_NAME}-${IMAGE_TAG}" \
  --machine-type="${MACHINE_TYPE}" --min-replica-count=1 --max-replica-count=3 --traffic-split=0=100

cat <<EOF
Deployed. Put these in app/validated_agents/registry.yaml (agents.echo_wall_motion.model):
  model_id: "${MODEL_ID}"
  endpoint_id: "${ENDPOINT_ID}"
  runtime_fingerprint / weights_sha256 / thresholds / red_flags: copy from model_card.json (output_adapter)
  study_prefix: "${STUDY_PREFIX}"
EOF
