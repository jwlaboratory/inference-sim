/* Defaults mirror config.py — the backend is the source of truth; anything
   posted here overrides config.py for that run only. */
export const DEFAULT_CONFIG = {
  config: {
    SEED: 42, NUM_REQUESTS: 400,
    DATASET: 'alessiotoniolo/ART-Chat-2.5M', DATASET_SPLIT: 'train',
    DATASET_OFFSET: -1, ARRIVAL_SCALE: 4, BLOCK_TOKENS: 256,
    PARAMS: 8e9, ACTIVE_PARAMS: 8e9, DTYPE_BYTES: 2,
    LAYERS: 32, KV_HEADS: 8, HEAD_DIM: 128,
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
const dense = (name, p, layers, kv, hd, dtype = 2) =>
  ({ name, PARAMS: p, ACTIVE_PARAMS: p, LAYERS: layers, KV_HEADS: kv, HEAD_DIM: hd, DTYPE_BYTES: dtype })
const moe = (name, total, active, layers, dtype) =>
  ({ name, PARAMS: total, ACTIVE_PARAMS: active, LAYERS: layers, KV_HEADS: 1, HEAD_DIM: 288, DTYPE_BYTES: dtype })
export const MODEL_PRESETS = [
  dense('Llama 3.2 1B',      1.24e9, 16, 8,  64),
  dense('Llama 3.2 3B',      3.21e9, 28, 8,  128),
  dense('Llama 3.1 8B',      8.03e9, 32, 8,  128),
  dense('Llama 3.3 70B',     70.6e9, 80, 8,  128),
  dense('Mistral 7B',        7.25e9, 32, 8,  128),
  dense('Mistral Small 24B', 23.6e9, 40, 8,  128),
  dense('Qwen2.5 7B',        7.62e9, 28, 4,  128),
  dense('Qwen2.5 72B',       72.7e9, 80, 8,  128),
  dense('Qwen3 8B',          8.19e9, 36, 8,  128),
  dense('Qwen3 32B',         32.8e9, 64, 8,  128),
  dense('Gemma 2 9B',        9.24e9, 42, 8,  256),
  dense('Gemma 2 27B',       27.2e9, 46, 16, 128),
  dense('Phi-4 14B',         14.7e9, 40, 10, 128),
  /* Frontier MoE models. These use MLA-style compressed KV caches (576
     elements/token/layer), expressed here in the sim's GQA formula as the
     equivalent 1 kv_head × 288 head_dim. DTYPE_BYTES is the native weight
     dtype (int4 QAT / MXFP4 / fp8 / bf16). */
  moe('DeepSeek-V4-Flash 284B', 284e9,  13e9, 43, 1),   // fp8, 6/256 experts
  moe('GLM-5.2 744B',           744e9,  40e9, 78, 2),   // bf16, 8/256 experts
  moe('Kimi K2.7 Code 1T',      1000e9, 32e9, 61, 0.5), // int4 QAT, 8/384 experts
  moe('DeepSeek-V4-Pro 1.6T',   1600e9, 49e9, 61, 1),   // fp8, 6/384 experts
  /* K3: layers & KV geometry unpublished until the weights drop (2026-07-27);
     layer count and KV shape are K2-family placeholders. MXFP4 weights. */
  moe('Kimi K3 2.8T (approx)',  2800e9, 50e9, 61, 0.5),
]
/* Keys a preset fills in; matching ignores DTYPE_BYTES so a quantized
   variant still reads as the same model in the dropdown. */
export const PRESET_KEYS = ['PARAMS', 'ACTIVE_PARAMS', 'LAYERS', 'KV_HEADS', 'HEAD_DIM', 'DTYPE_BYTES']
export const PRESET_MATCH_KEYS = PRESET_KEYS.filter((k) => k !== 'DTYPE_BYTES')

/* Quantization = bytes per weight/KV element (applied on top of any preset). */
export const QUANT_OPTIONS = [
  ['fp16 / bf16', 2], ['fp8 / int8', 1], ['int4', 0.5],
]

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
    ['PARAMS', 'params (B)', 1e9, 1], ['ACTIVE_PARAMS', 'active params (B)', 1e9, 1],
    ['LAYERS', 'layers', 1, 1],
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
export const TIER_DESC = {
  hbm: 'Prefix KV cache was already in the serving GPU’s HBM — fastest case, no data movement needed before decode.',
  ram: 'Prefix KV cache was found in the host’s CPU RAM and copied to the GPU over PCIe before prefill.',
  rdma: 'Prefix KV cache lived on another GPU in the cluster and was pulled over the RDMA network.',
  disk: 'Prefix KV cache was loaded from local disk (NVMe) — slowest cache tier, but still cheaper than recomputing.',
  miss: 'No cached prefix found anywhere — the full prompt had to be prefilled (recomputed) from scratch.',
}
export const QUEUE_DESC =
  'The gray strip under each GPU lane shows how many requests were waiting in that GPU’s queue over time — darker means a deeper backlog.'

export const fmtT = (s) =>
  s >= 3600 ? `${(s / 3600).toFixed(1)}h`
  : s >= 60 ? `${Math.floor(s / 60)}m${String(Math.floor(s % 60)).padStart(2, '0')}s`
  : `${s.toFixed(s < 10 ? 1 : 0)}s`
