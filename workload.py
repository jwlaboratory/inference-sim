"""Workload: replay a window of a real inference trace from Hugging Face.

The dataset must be Mooncake-format (e.g. alessiotoniolo/ART-Chat-2.5M):
each row has timestamp_ms, input_length, output_length, and hash_ids —
hashes of consecutive BLOCK_TOKENS-sized prompt blocks. Two requests share
a prefix exactly when their leading hash_ids match, so the simulator's
caches operate on the recorded hashes directly, no tokenization needed.

A start row is chosen (DATASET_OFFSET, or a random point derived from SEED)
and NUM_REQUESTS consecutive rows are replayed, arrival gaps scaled by
ARRIVAL_SCALE. Rows are fetched via the HF datasets-server rows API and
cached in memory, so re-runs over the same window are free.
"""
import json
import math
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from types import SimpleNamespace

import config

API = "https://datasets-server.huggingface.co"
_cache = {}


@dataclass
class Request:
    id: int
    arrival: float
    group: str               # system-prompt / user hash, for display only
    prefix_tokens: int       # block-covered prompt tokens (the cacheable part)
    input_tokens: int
    output_tokens: int
    blocks: list = field(default_factory=list)        # prompt block hashes
    cache_blocks: list = field(default_factory=list)  # blocks resident after serving


def _get(url):
    if url not in _cache:
        with urllib.request.urlopen(url, timeout=60) as resp:
            _cache[url] = json.load(resp)
    return _cache[url]


def _total_rows(dataset):
    return _get(f"{API}/size?dataset={urllib.parse.quote(dataset)}")["size"]["dataset"]["num_rows"]


def _rows(dataset, split, offset, n):
    rows = []
    while len(rows) < n:
        url = (f"{API}/rows?dataset={urllib.parse.quote(dataset)}&config=default"
               f"&split={split}&offset={offset + len(rows)}&length={min(100, n - len(rows))}")
        rows += [x["row"] for x in _get(url)["rows"]]
    return rows


def generate(cfg=None):
    c = cfg or SimpleNamespace(**config.as_dict())
    random.seed(c.SEED)
    n = int(c.NUM_REQUESTS)
    total = _total_rows(c.DATASET)
    offset = int(c.DATASET_OFFSET) if c.DATASET_OFFSET >= 0 \
        else random.randrange(max(1, total - n))
    rows = _rows(c.DATASET, c.DATASET_SPLIT, offset, n)

    t0 = rows[0]["timestamp_ms"]
    reqs = []
    for row in rows:
        inp = max(1, int(row["input_length"]))
        out = max(1, int(row["output_length"]))
        blocks = list(row["hash_ids"] or [])
        # the response extends the cached context past the recorded blocks
        extra = math.ceil((inp + out) / c.BLOCK_TOKENS) - len(blocks)
        cache_blocks = blocks + [f"{row['request_id']}#o{j}" for j in range(max(0, extra))]
        group = (row.get("system_prompt_hash") or row.get("token_hash") or "anon")[:10]
        reqs.append(Request(0, (row["timestamp_ms"] - t0) / 1000 * c.ARRIVAL_SCALE,
                            group, min(len(blocks) * c.BLOCK_TOKENS, inp), inp, out,
                            blocks, cache_blocks))

    reqs.sort(key=lambda r: r.arrival)
    for i, r in enumerate(reqs):
        r.id = i
    print(f"replaying {len(reqs)} requests from {c.DATASET}[{offset}:{offset + n}]")
    return reqs


if __name__ == "__main__":
    reqs = generate()
    mean_in = sum(r.input_tokens for r in reqs) / len(reqs)
    mean_out = sum(r.output_tokens for r in reqs) / len(reqs)
    print(f"span={reqs[-1].arrival:.0f}s  mean_input={mean_in:.0f}  mean_output={mean_out:.0f}")
