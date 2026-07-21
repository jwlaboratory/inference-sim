"""FastAPI backend: serves the built React UI and runs the sim for posted configs.

Run:  python3 server.py   (or: uvicorn server:app --reload)
Then open http://localhost:8000

UI development:  cd ui && npm run dev   (Vite dev server, proxies /api here)
UI release:      cd ui && npm run build  (this server serves ui/dist)

POST /api/simulate body (all parts optional; defaults come from config.py):
  {
    "config":  { "NUM_REQUESTS": 400, "MFU": 0.5, ... },     # any config.py name
    "specs":   { "H100": { "flops": ..., "hbm_bw": ..., ... } },
    "cluster": [ { "name": "node0", "spec": "H100", "gpus": 8 }, ... ]
  }
Returns 400 with a message when the model doesn't fit a node's combined HBM.
"""
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

import config
from router import POLICIES
from simulate import run
from workload import generate

app = FastAPI(title="inference-sim")


@app.post("/api/simulate")
def simulate(body: dict):
    cfg_dict = config.as_dict()
    cfg_dict.update(body.get("config") or {})

    specs = {k: v for k, v in cfg_dict.items() if isinstance(v, config.GPUSpec)}
    for name, s in (body.get("specs") or {}).items():
        specs[name] = config.GPUSpec(name=name,
                                     **{k: float(v) for k, v in s.items() if k != "name"})
    if body.get("cluster"):
        cfg_dict["CLUSTER"] = [(g["name"], specs[g["spec"]], int(g.get("gpus", 1)))
                               for g in body["cluster"]]

    cfg = SimpleNamespace(**cfg_dict)
    requests = generate(cfg)
    try:
        runs = {policy: run(policy, requests, cfg) for policy in POLICIES}
    except ValueError as e:      # model doesn't fit / request exceeds KV headroom
        raise HTTPException(status_code=400, detail=str(e))
    return {"requests": {"count": len(requests),
                         "mean_input": sum(r.input_tokens for r in requests) / len(requests),
                         "mean_output": sum(r.output_tokens for r in requests) / len(requests)},
            "runs": runs}


# mounted last so /api routes take precedence
app.mount("/", StaticFiles(directory=Path(__file__).parent / "ui" / "dist", html=True))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
