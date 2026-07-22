"""Guard against the UI defaults drifting from config.py.

ui/src/config.js hand-mirrors config.py (the backend is the source of truth,
but UI-posted values override it per run — so a stale mirror silently changes
what the site simulates). This test imports both real sources — config.py as a
module, and ui/src/config.js via Node — and asserts every scalar tunable, GPU
spec, and cluster node matches. No regex/text parsing: both sides are executed.

Run directly (`python3 test_config_sync.py`) or under pytest. Needs `node` on
PATH (the frontend requires it anyway).
"""
import json
import math
import subprocess
from dataclasses import fields
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent
UI_CONFIG_JS = ROOT / "ui" / "src" / "config.js"
SPEC_FIELDS = [f.name for f in fields(config.GPUSpec) if f.name != "name"]


def load_ui_defaults():
    """DEFAULT_CONFIG from ui/src/config.js, as plain Python data."""
    js = (
        f"import({json.dumps(UI_CONFIG_JS.as_uri())})"
        ".then(m => process.stdout.write(JSON.stringify(m.DEFAULT_CONFIG)))"
    )
    out = subprocess.run(
        ["node", "-e", js], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)


def eq(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
    return a == b


def backend_scalars():
    """Scalar tunables in config.py: the UPPER-case globals that aren't the
    GB helper, a GPUSpec, or the CLUSTER (those are checked separately)."""
    return {
        k: v for k, v in config.as_dict().items()
        if k != "GB" and k != "CLUSTER" and not isinstance(v, config.GPUSpec)
    }


def check():
    ui = load_ui_defaults()
    errors = []

    # --- scalar tunables (both directions: catches added/removed keys) ---
    backend = backend_scalars()
    ui_scalars = ui["config"]
    for k in sorted(set(backend) | set(ui_scalars)):
        if k not in ui_scalars:
            errors.append(f"config[{k!r}]: missing from ui/src/config.js (config.py = {backend[k]!r})")
        elif k not in backend:
            errors.append(f"config[{k!r}]: present in UI but not config.py (UI = {ui_scalars[k]!r})")
        elif not eq(backend[k], ui_scalars[k]):
            errors.append(f"config[{k!r}]: config.py = {backend[k]!r} vs UI = {ui_scalars[k]!r}")

    # --- GPU specs ---
    backend_specs = {k: v for k, v in config.as_dict().items()
                     if isinstance(v, config.GPUSpec)}
    ui_specs = ui["specs"]
    for name in sorted(set(backend_specs) | set(ui_specs)):
        if name not in ui_specs:
            errors.append(f"spec {name!r}: missing from ui/src/config.js")
        elif name not in backend_specs:
            errors.append(f"spec {name!r}: present in UI but not config.py")
        else:
            for f in SPEC_FIELDS:
                bv, uv = getattr(backend_specs[name], f), ui_specs[name].get(f)
                if not eq(bv, uv):
                    errors.append(f"spec {name}.{f}: config.py = {bv!r} vs UI = {uv!r}")

    # --- cluster ---
    backend_cluster = config.CLUSTER
    ui_cluster = ui["cluster"]
    if len(backend_cluster) != len(ui_cluster):
        errors.append(f"cluster length: config.py = {len(backend_cluster)} vs UI = {len(ui_cluster)}")
    for i, (bnode, unode) in enumerate(zip(backend_cluster, ui_cluster)):
        bname, bspec, bgpus = bnode
        if bname != unode.get("name"):
            errors.append(f"cluster[{i}].name: config.py = {bname!r} vs UI = {unode.get('name')!r}")
        if bspec.name != unode.get("spec"):
            errors.append(f"cluster[{i}].spec: config.py = {bspec.name!r} vs UI = {unode.get('spec')!r}")
        if not eq(bgpus, unode.get("gpus")):
            errors.append(f"cluster[{i}].gpus: config.py = {bgpus!r} vs UI = {unode.get('gpus')!r}")

    return errors


def test_ui_defaults_match_config():
    errors = check()
    assert not errors, "UI defaults drifted from config.py:\n  " + "\n  ".join(errors)


if __name__ == "__main__":
    import sys

    errs = check()
    if errs:
        print("FAIL: ui/src/config.js drifted from config.py:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print("OK: ui/src/config.js defaults match config.py")
