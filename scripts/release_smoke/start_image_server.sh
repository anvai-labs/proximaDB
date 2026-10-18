#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start the RELEASED server image as the target the SDK smokes talk to, and
# wait until it is healthy AND reports the release version.
#
# Usage: start_image_server.sh <image_repo> <version> <host_port> [container_name]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:?image repo}"; VERSION="${2:?version}"; PORT="${3:?host port}"
NAME="${4:-proximadb-release-smoke}"
IMAGE="${REPO}:${VERSION}"
bash "$HERE/retry.sh" 5 30 docker pull "$IMAGE"
docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:5678" "$IMAGE"
if ! bash "$HERE/wait_healthy.sh" "http://127.0.0.1:${PORT}" "$VERSION" 180; then
  docker logs --tail 200 "$NAME"
  exit 1
fi
