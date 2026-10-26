#!/usr/bin/env python3
"""Bounded, task-owned WSL direct-I/O diagnostic; not an EBS benchmark."""
import concurrent.futures
import json
import mmap
import os
from pathlib import Path
import random
import statistics
import time

ROOT = Path('/tmp/proximadb-storage-class-review/media-results')
ROOT.mkdir(parents=True, exist_ok=True)
FILE = ROOT / 'owned-direct-io.bin'
MIB = 1024 * 1024
SIZE = 1024 * MIB
result = {'kind': 'WSL/VHD direct I/O diagnostic, host cache not controlled',
          'file_bytes': SIZE, 'arms': []}
fd = os.open(FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_DIRECT, 0o600)
try:
    with mmap.mmap(-1, 8 * MIB) as data:
        data[:] = os.urandom(8 * MIB)
        for offset in range(0, SIZE, len(data)):
            written = os.pwritev(fd, [data], offset)
            if written != len(data):
                raise RuntimeError('short preparation write')
    os.fsync(fd)
    for qd in (1, 8):
        for size_mib in (4, 8, 16, 32, 40, 64):
            size = size_mib * MIB
            for pattern in ('sequential', 'random'):
                offsets = list(range(0, SIZE - size + 1, size))
                if pattern == 'random':
                    random.Random(97025).shuffle(offsets)
                offsets = offsets[:min(64, len(offsets))]
                # Reuse one aligned buffer per worker. Buffer allocation is outside
                # each timed operation; wall throughput includes scheduling overhead.
                chunks = [offsets[i::qd] for i in range(qd)]
                def worker(items):
                    latencies = []
                    with mmap.mmap(-1, size) as buf:
                        for off in items:
                            started = time.perf_counter()
                            got = os.preadv(fd, [buf], off)
                            latencies.append((time.perf_counter()-started)*1000)
                            if got != size:
                                raise RuntimeError('short diagnostic read')
                    return latencies
                started = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=qd) as pool:
                    samples = sum(pool.map(worker, chunks), [])
                elapsed = time.perf_counter()-started
                ordered = sorted(samples)
                arm = {'qd': qd, 'range_mib': size_mib, 'pattern': pattern,
                       'reads': len(samples), 'mib_per_s': len(samples)*size_mib/elapsed,
                       'p50_ms': statistics.median(samples),
                       'p95_ms': ordered[min(len(ordered)-1, int(.95*len(ordered)))]}
                result['arms'].append(arm)
                print(json.dumps(arm), flush=True)
finally:
    os.close(fd)
    # Only the test file created with O_EXCL above is removed.
    FILE.unlink()
    (ROOT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
