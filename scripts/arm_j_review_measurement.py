import json, sys, time
import numpy as np
from cviaf.lab.compare import load_registry, data_axis, _score_data_axis

corpus, out = sys.argv[1], sys.argv[2]
t0 = time.time()
registry = load_registry(corpus)
print(f"{len(registry)} models", flush=True)
rows = data_axis(registry, alpha=0.05, log=lambda s: print(s, flush=True))
res = {"corpus": corpus, "n_models": len(registry), "alpha": 0.05,
       "data_axis": {"scores": _score_data_axis(rows), "per_model": rows},
       "elapsed_seconds": round(time.time()-t0, 1)}
class Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        return str(o)
json.dump(res, open(out, "w"), indent=1, cls=Enc)
print("wrote", out, "in", res["elapsed_seconds"], "s", flush=True)
