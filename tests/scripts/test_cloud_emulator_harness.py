#!/usr/bin/env python3
"""Regression checks for immutable cloud-emulator container inputs."""

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "scripts/run_cloud_emulator_tests.sh"


class CloudEmulatorHarnessTest(unittest.TestCase):
    def test_minio_uses_a_quay_digest_not_docker_hub_latest(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        image = re.search(r'^MINIO_IMAGE="([^"]+)"$', source, re.MULTILINE)
        self.assertIsNotNone(image, "the shared harness must declare MINIO_IMAGE")
        reference = image.group(1) if image else ""
        self.assertRegex(
            reference,
            r"^quay\.io/minio/minio@sha256:[0-9a-f]{64}$",
            "the MinIO emulator must be immutable and independent of Docker Hub tags",
        )
        self.assertIn('"$MINIO_IMAGE" server /data', source)
        self.assertNotIn("minio/minio server /data", source)
        for binding in (
            "127.0.0.1:10000:10000",
            "127.0.0.1:9000:9000",
            "127.0.0.1:4443:4443",
        ):
            self.assertIn(
                binding,
                source,
                "emulators with development credentials must bind only to loopback",
            )
        self.assertIn("/minio/health/ready", source)
        self.assertIn("create_minio_bucket", source)
        self.assertNotIn('(minio bucket exists)', source)


if __name__ == "__main__":
    unittest.main()
