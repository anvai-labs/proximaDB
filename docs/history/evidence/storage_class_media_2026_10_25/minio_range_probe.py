#!/usr/bin/env python3
"""Task-owned MinIO range diagnostic, not a ProximaDB query benchmark."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import random
import statistics
import time
import uuid

import boto3
from botocore.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minio-pid', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    credentials = {}
    if args.minio_pid:
        # Reuse only the explicitly selected local service's credentials in memory.
        # Never log them or save them in the measurement artifact.
        env = dict(item.split(b'=', 1) for item in
                   Path(f'/proc/{args.minio_pid}/environ').read_bytes().split(b'\0')
                   if b'=' in item)
        user = env.get(b'MINIO_ROOT_USER')
        password = env.get(b'MINIO_ROOT_PASSWORD')
        if not user or not password:
            raise RuntimeError('Selected service has no root credentials in its environment')
        credentials = dict(aws_access_key_id=user.decode(),
                           aws_secret_access_key=password.decode())
    clients = [boto3.client('s3', endpoint_url=url, region_name='us-east-1',
               config=Config(s3={'addressing_style': 'path'},
                             retries={'total_max_attempts': 1}, max_pool_connections=8),
               **credentials)
               for url in ('http://127.0.0.1:12000', 'http://127.0.0.1:12002')]
    bucket = 'seam-review-' + uuid.uuid4().hex[:16]
    key = 'owned-range-probe.bin'
    mib = 1024 * 1024
    size = 256 * mib
    result = {'kind': 'MinIO range diagnostic; warm host caches; no ANN query',
              'date': '2026-10-25', 'object_bytes': size,
              'client': 'boto3 streamed 1 MiB reads; QD1 and QD8', 'arms': []}
    created = False
    upload_id = None
    try:
        clients[0].create_bucket(Bucket=bucket)
        created = True
        upload_id = clients[0].create_multipart_upload(Bucket=bucket, Key=key)['UploadId']
        block = os.urandom(8 * mib)
        parts = []
        for n in range(1, 33):
            response = clients[0].upload_part(Bucket=bucket, Key=key, UploadId=upload_id,
                                              PartNumber=n, Body=block)
            parts.append({'PartNumber': n, 'ETag': response['ETag']})
        clients[0].complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id,
                                            MultipartUpload={'Parts': parts})
        upload_id = None
        for ci, client in enumerate(clients):
            for qd in (1, 8):
                for range_mib in (4, 8, 16, 32, 40, 64):
                    length = range_mib * mib
                    # Same seeded noncontiguous offsets for every endpoint/QD arm.
                    rng = random.Random(97025)
                    offsets = [rng.randrange((size-length)//mib + 1)*mib for _ in range(8)]
                    def read(offset):
                        started = time.perf_counter()
                        response = client.get_object(Bucket=bucket, Key=key,
                                     Range=f'bytes={offset}-{offset+length-1}')
                        count = 0
                        with response['Body'] as body:
                            while chunk := body.read(mib):
                                count += len(chunk)
                        if count != length or response['ResponseMetadata']['HTTPStatusCode'] != 206:
                            raise RuntimeError('Invalid ranged response')
                        return (time.perf_counter()-started)*1000
                    started = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=qd) as pool:
                        samples = list(pool.map(read, offsets))
                    elapsed = time.perf_counter()-started
                    arm = {'endpoint': client.meta.endpoint_url, 'proxy': bool(ci),
                           'qd': qd, 'range_mib': range_mib, 'reads': len(samples),
                           'samples_ms': samples, 'p50_ms': statistics.median(samples),
                           'max_ms': max(samples), 'mib_per_s': len(samples)*range_mib/elapsed}
                    result['arms'].append(arm)
                    print(json.dumps({k: v for k, v in arm.items() if k != 'samples_ms'}), flush=True)
    finally:
        if upload_id:
            clients[0].abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        if created:
            clients[0].delete_object(Bucket=bucket, Key=key)
            clients[0].delete_bucket(Bucket=bucket)
        args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
