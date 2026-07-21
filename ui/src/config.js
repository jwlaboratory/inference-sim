/* Defaults mirror config.py — the backend is the source of truth; anything
   posted here overrides config.py for that run only. */
export const DEFAULT_CONFIG = {
  config: {
    SEED: 42, NUM_REQUESTS: 400,
    DATASET: 'alessiotoniolo/ART-Chat-2.5M', DATASET_SPLIT: 'train',
    DATASET_OFFSET: -1, ARRIVAL_SCALE: 4, BLOCK_TOKENS: 256,
    PARAMS: 8e9, DTYPE_BYTES: 2, LAYERS: 32, KV_HEADS: 8, HEAD_DIM: 128,
    MFU: 0.5, MBU: 0.8,
    IMBALANCE_ABS: 2, IMBALANCE_REL: 1.5, DISK_CACHE: true,
  },
  specs: {
    H100: { flops: 989e12, hbm_bw: 3.35e12, hbm_cap: 80e9, ram_bw: 55e9, rdma_bw: 50e9, disk_bw: 7e9 },
    A100: { flops: 312e12, hbm_bw: 2.00e12, hbm_cap: 80e9, ram_bw: 25e9, rdma_bw: 25e9, disk_bw: 5e9 },
  },
  cluster: [
    { name: 'gpu0', spec: 'H100' }, { name: 'gpu1', spec: 'H100' },
    { name: 'gpu2', spec: 'H100' }, { name: 'gpu3', spec: 'H100' },
  ],
}

/* Architecture specs from each model's Hugging Face config.json (fp16/bf16
   weights). head_dim = hidden_size / num_attention_heads unless the config
   sets it explicitly. Selecting a preset fills the Model fields below. */
export const MODEL_PRESETS = [
  { name: 'Llama 3.2 1B',        PARAMS: 1.24e9, LAYERS: 16, KV_HEADS: 8,  HEAD_DIM: 64,  DTYPE_BYTES: 2 },
  { name: 'Llama 3.2 3B',        PARAMS: 3.21e9, LAYERS: 28, KV_HEADS: 8,  HEAD_DIM: 128, DTYPE_BYTES: 2 },
  { name: 'Llama 3.1 8B',        PARAMS: 8.03e9, LAYERS: 32, KV_HEADS: 8,  HEAD_DIM: 128, DTYPE_BYTES: 2 },
  { name: 'Llama 3.3 70B',       PARAMS: 70.6e9, LAYERS: 80, KV_HEADS: 8,  HEAD_DIM: 128, DTYPE_BYTES: 2 },
  { name: 'Mistral 7B',          PARAMS: 7.25e9, LAYERS: 32, KV_HEADS: 8,  HEAD_DIM: 128, DTYPE_BYTES: 2 },
  { name: 'Mistral Small 24B',   PARAMS: 23.6e9, LAYERS: 40, KV_HEADS: 8,  HEAD_DIM: 128, DTYPE_BYTES: 2 },
  { name: 'Qwen2.5 7B',          PARAMS: 7.62e9, LAYERS: 28, KV_HEADS: 4,  HEAD_DIM: 128, DTYPE_BYTES: 2 },
  { name: 'Qwen2.5 72B',         PARAMS: 72.7e9, LAYERS: 80, KV_HEADS: 8,  HEAD_DIM: 128, DTYPE_BYTES: 2 },
  { name: 'Qwen3 8B',            PARAMS: 8.19e9, LAYERS: 36, KV_HEADS: 8,  HEAD_DIM: 128, DTYPE_BYTES: 2 },
  { name: 'Qwen3 32B',           PARAMS: 32.8e9, LAYERS: 64, KV_HEADS: 8,  HEAD_DIM: 128, DTYPE_BYTES: 2 },
  { name: 'Gemma 2 9B',          PARAMS: 9.24e9, LAYERS: 42, KV_HEADS: 8,  HEAD_DIM: 256, DTYPE_BYTES: 2 },
  { name: 'Gemma 2 27B',         PARAMS: 27.2e9, LAYERS: 46, KV_HEADS: 16, HEAD_DIM: 128, DTYPE_BYTES: 2 },
  { name: 'Phi-4 14B',           PARAMS: 14.7e9, LAYERS: 40, KV_HEADS: 10, HEAD_DIM: 128, DTYPE_BYTES: 2 },
]
export const PRESET_KEYS = ['PARAMS', 'LAYERS', 'KV_HEADS', 'HEAD_DIM', 'DTYPE_BYTES']

/* Field schema: [config key, label, scale, step]. Shown value = raw / scale. */
export const SECTIONS = [
  ['Trace replay', [
    ['NUM_REQUESTS', 'requests (window)', 1, 50],
    ['DATASET_OFFSET', 'start row (-1 = random)', 1, 1000],
    ['ARRIVAL_SCALE', 'arrival scale', 1, 0.5],
    ['SEED', 'seed', 1, 1],
    ['BLOCK_TOKENS', 'tokens / block', 1, 64],
  ]],
  ['Model', [
    ['PARAMS', 'params (B)', 1e9, 1], ['LAYERS', 'layers', 1, 1],
    ['KV_HEADS', 'kv heads', 1, 1], ['HEAD_DIM', 'head dim', 1, 16],
    ['MFU', 'MFU', 1, 0.05], ['MBU', 'MBU', 1, 0.05],
  ]],
  ['Router', [
    ['IMBALANCE_ABS', 'imbalance abs s', 1, 0.5], ['IMBALANCE_REL', 'imbalance rel', 1, 0.1],
  ]],
]

export const SPEC_FIELDS = [
  ['flops', 'TFLOPS', 1e12], ['hbm_bw', 'HBM GB/s', 1e9], ['hbm_cap', 'HBM GB', 1e9],
  ['ram_bw', 'PCIe GB/s', 1e9], ['rdma_bw', 'RDMA GB/s', 1e9], ['disk_bw', 'disk GB/s', 1e9],
]

export const POLICY_NAMES = ['cache_aware', 'least_load', 'round_robin', 'random']

/* Marks are colored by where the request's prefix was served from. */
export const TIERS = ['hbm', 'ram', 'rdma', 'disk', 'miss']
export const TIER_COLOR = {
  hbm: 'var(--hbm)', ram: 'var(--ram)', rdma: 'var(--rdma)',
  disk: 'var(--disk)', miss: 'var(--miss)',
}

export const fmtT = (s) =>
  s >= 3600 ? `${(s / 3600).toFixed(1)}h`
  : s >= 60 ? `${Math.floor(s / 60)}m${String(Math.floor(s % 60)).padStart(2, '0')}s`
  : `${s.toFixed(s < 10 ? 1 : 0)}s`
