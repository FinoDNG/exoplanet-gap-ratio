"""Part des rapports proches de 1 (r > 0,8) chez les planètes, avec intervalle à 95 % par rééchantillonnage des systèmes."""
import json, numpy as np
rng = np.random.default_rng(5)
S = [s for s in json.load(open("systemes.json")) if len(s["planetes"]) >= 3]
S = [s for s in S if np.min(np.diff(np.log([p["periode"] for p in s["planetes"]]))) >= np.log(1.1)]
def rap(x):
    s = np.diff(np.sort(x)); return np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
g = [rap(np.log([p["periode"] for p in s["planetes"]])) for s in S]
k = len(g); f = lambda r: np.mean(r > 0.8) * 100
v = [f(np.concatenate([g[i] for i in rng.integers(0, k, k)])) for _ in range(4000)]
print("r > 0,8 : planètes %.1f %%, IC95 [%.1f ; %.1f]" % (f(np.concatenate(g)), *np.percentile(v, [2.5, 97.5])))
z = np.load("zeros.npy") ; print("zéros : %.1f %%" % (np.mean(rap(z) > 0.8) * 100))
