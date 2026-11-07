"""Éparpillement des planètes comparé à celui des nombres premiers et des zéros de zêta.
Statistique principale : rapport d'écarts consécutifs r = min(s1, s2) / max(s1, s2)
(Oganesyan–Huse ; valeurs théoriques Poisson 2 ln 2 − 1 ≈ 0,386, GUE ≈ 0,600 d'après Atas et al. 2013).
Ce rapport ne dépend ni de l'échelle ni de la densité locale : aucune normalisation n'est nécessaire."""
import json, numpy as np, mpmath
rng = np.random.default_rng(20261103)
S = json.load(open("systemes.json"))

def rapports(x):
    s = np.diff(np.sort(np.asarray(x, float)))
    a, b = s[:-1], s[1:]
    return np.minimum(a, b) / np.maximum(a, b)

def boot(groupes, n=4000):
    """moyenne de r et intervalle à 95 % par rééchantillonnage des systèmes"""
    allr = np.concatenate(groupes); m = allr.mean()
    k = len(groupes); bs = []
    for _ in range(n):
        pick = rng.integers(0, k, k)
        bs.append(np.concatenate([groupes[i] for i in pick]).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return m, lo, hi, len(allr), k

# ---------- planètes
def sel(cond):
    return [s for s in S if len(s["planetes"]) >= 3 and cond(s)]
jeux = {
    "toutes (≥ 3 planètes)": sel(lambda s: True),
    "≥ 4 planètes": sel(lambda s: len(s["planetes"]) >= 4),
    "découvertes par transit uniquement": sel(lambda s: all(p["methode"] == "transit" for p in s["planetes"])),
    "systèmes Kepler": sel(lambda s: s["systeme"].startswith("Kepler-")),
    "vitesse radiale uniquement": sel(lambda s: all(p["methode"] == "RV" for p in s["planetes"])),
}
res = {}
for nom, js in jeux.items():
    g = [rapports(np.log([p["periode"] for p in s["planetes"]])) for s in js]
    res[nom] = boot(g)
# Système solaire (périodes sidérales en jours)
sol = [87.969, 224.701, 365.256, 686.980, 4332.59, 10759.22, 30688.5, 60182.0]
r_sol = rapports(np.log(sol))
res["Système solaire (6 rapports)"] = (r_sol.mean(), np.nan, np.nan, 6, 1)

# ---------- références (même statistique)
N = 10**8
s_ = np.ones(N // 2, dtype=bool); s_[0] = False
for i in range(1, int(N**0.5) // 2 + 1):
    if s_[i]:
        p = 2 * i + 1; s_[p * p // 2::p] = False
P = (np.flatnonzero(s_).astype(np.int64) * 2 + 1); del s_
for lo_, hi_ in ((10**4, 10**6), (10**6, 10**8)):
    q = P[(P > lo_) & (P < hi_)]
    r = rapports(q); res[f"nombres premiers entre {lo_:.0e} et {hi_:.0e}"] = (r.mean(), np.nan, np.nan, len(r), None)
    res[f"  dont part de r = 1 exactement (écarts égaux)"] = (float(np.mean(r == 1)), np.nan, np.nan, len(r), None)
Zs = np.load("zeros.npy")
Zs[:200] = [float(mpmath.zetazero(n).imag) for n in range(1, 201)]
rz = rapports(Zs); res["zéros de zêta (5 598 premiers)"] = (rz.mean(), np.nan, np.nan, len(rz), None)
# Poisson et GUE simulés
rp = rapports(np.cumsum(rng.exponential(size=2_000_000))); res["hasard pur (Poisson, simulé)"] = (rp.mean(), np.nan, np.nan, len(rp), None)
rg = []
for _ in range(400):
    A = rng.normal(size=(200, 200)) + 1j * rng.normal(size=(200, 200)); H = (A + A.conj().T) / 2
    ev = np.linalg.eigvalsh(H); rg.append(rapports(ev[50:150]))
rg = np.concatenate(rg); res["matrices GUE (simulé)"] = (rg.mean(), np.nan, np.nan, len(rg), None)
rr = rapports(np.arange(1000.0)); res["peigne parfait (écarts égaux)"] = (rr.mean(), np.nan, np.nan, len(rr), None)

for k, (m, lo, hi, n, ks) in res.items():
    ic = f"[{lo:.3f} ; {hi:.3f}]" if not np.isnan(lo) else ""
    print(f"{k:45s} <r> = {m:.4f} {ic:20s} n = {n}" + (f"  systèmes = {ks}" if ks else ""))

# ---------- distribution de r : planètes, premiers, zéros, Poisson, GUE
bins = np.linspace(0, 1, 6)
allpl = np.concatenate([rapports(np.log([p["periode"] for p in s["planetes"]])) for s in jeux["toutes (≥ 3 planètes)"]])
qq = P[(P > 10**6)]; rq = rapports(qq)
print("\nrépartition de r par tranches de 0,2 :")
for nom, r in (("planètes", allpl), ("premiers", rq), ("zéros", rz), ("Poisson", rp), ("GUE", rg)):
    h = np.histogram(r, bins)[0] / len(r) * 100
    print(f"{nom:10s}", "  ".join(f"{x:5.1f}%" for x in h))
print("petits rapports r < 0,1 :", {nom: round(float(np.mean(r < 0.1) * 100), 1) for nom, r in
      (("planètes", allpl), ("premiers", rq), ("zéros", rz), ("Poisson", rp), ("GUE", rg))})

# ---------- « petits pois dans une cosse » : tailles et espacements voisins
pairs, sp = [], []
for s in jeux["toutes (≥ 3 planètes)"]:
    pl = s["planetes"]
    R = [p["rayon"] for p in pl]
    for i in range(len(pl) - 1):
        if R[i] and R[i + 1] and R[i] > 0 and R[i + 1] > 0: pairs.append((np.log(R[i]), np.log(R[i + 1])))
    lp = np.log([p["periode"] for p in pl]); d = np.diff(lp)
    for i in range(len(d) - 1): sp.append((d[i], d[i + 1]))
pairs = np.array(pairs); sp = np.array(sp)
def corr_perm(a, n=20000):
    c = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
    pool = a.ravel(); cs = []
    for _ in range(n):
        x = rng.permutation(pool); cs.append(np.corrcoef(x[::2], x[1::2])[0, 1])
    return c, float(np.mean(np.abs(cs) >= abs(c)))
c1, p1 = corr_perm(pairs); c2, p2 = corr_perm(sp)
print(f"\ntailles de planètes voisines : corrélation {c1:.3f} (paires = {len(pairs)}), p = {p1:.5f}")
print(f"espacements voisins : corrélation {c2:.3f} (paires = {len(sp)}), p = {p2:.5f}")
print("rapport de périodes voisines le plus petit (planètes) :", round(float(np.exp(np.min(np.concatenate([np.diff(np.log([p['periode'] for p in s['planetes']])) for s in jeux['toutes (≥ 3 planètes)']])))), 3))
json.dump({k: [float(x) if x is not None else None for x in v] for k, v in res.items()}, open("resultats.json", "w"), indent=1, ensure_ascii=False)
