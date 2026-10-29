#!/usr/bin/env bash
set -euo pipefail

exec > "$HOME/start_docker.log" 2>&1

AWS_REGION="ap-southeast-2"
ECR_REGISTRY="541448418548.dkr.ecr.ap-southeast-2.amazonaws.com"
IMAGE_NAME="yt-sentiment-comment"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG_FILE="$SCRIPT_DIR/../image-tag.txt"
CONTAINER_NAME="campusx-app"
HOST_PORT="80"
CONTAINER_PORT="5000"

if [[ ! -f "$IMAGE_TAG_FILE" ]]; then
    echo "Missing deployment image tag: $IMAGE_TAG_FILE"
    exit 1
fi

IMAGE_TAG="$(< "$IMAGE_TAG_FILE")"
if [[ ! "$IMAGE_TAG" =~ ^[[:xdigit:]]{40}$ ]]; then
    echo "Invalid deployment image tag: $IMAGE_TAG"
    exit 1
fi

IMAGE="$ECR_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"

sudo systemctl start docker
aws ecr get-login-password --region "$AWS_REGION" \
    | sudo docker login --username AWS --password-stdin "$ECR_REGISTRY"
sudo docker pull "$IMAGE"

if sudo docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    sudo docker rm --force "$CONTAINER_NAME"
fi

sudo docker run --detach \
    --restart unless-stopped \
    --publish "$HOST_PORT:$CONTAINER_PORT" \
    --name "$CONTAINER_NAME" \
    "$IMAGE"

for attempt in {1..30}; do
    if curl --fail --silent http://127.0.0.1/ >/dev/null; then
        echo "Deployment is healthy: $IMAGE"
        exit 0
    fi
    sleep 2
done

echo "Deployed container failed its HTTP health check"
sudo docker logs "$CONTAINER_NAME"
exit 1
