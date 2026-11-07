import json, numpy as np
rng = np.random.default_rng(7)
S = [s for s in json.load(open("systemes.json")) if len(s["planetes"]) >= 3]
def rapports(x):
    s = np.diff(np.sort(np.asarray(x, float))); a, b = s[:-1], s[1:]
    return np.minimum(a, b) / np.maximum(a, b)
def boot_mean(groupes, n=4000):
    k = len(groupes); bs = [np.concatenate([groupes[i] for i in rng.integers(0, k, k)]).mean() for _ in range(n)]
    return np.concatenate(groupes).mean(), np.percentile(bs, [2.5, 97.5]), bs
lp = [np.log([p["periode"] for p in s["planetes"]]) for s in S]
# 1) sans les 2 systèmes suspects (rapport de périodes < 1,1 : planètes probablement autour de deux étoiles différentes)
ok = [x for x in lp if np.min(np.exp(np.diff(x))) >= 1.1]
m, ic, bs = boot_mean([rapports(x) for x in ok])
print(f"sans paires < 1,1 : <r> = {m:.4f}  IC95 [{ic[0]:.3f} ; {ic[1]:.3f}]  systèmes = {len(ok)}")
bs = np.array(bs)
for nom, ref in (("GUE", 0.5996), ("zéros calculés", 0.6161), ("premiers 1e6-1e8", 0.4472), ("Poisson", 0.3863)):
    print(f"  part des rééchantillons au-dessous de {nom} ({ref}) : {np.mean(bs <= ref) * 100:.2f} %")
# 2) modèle « hasard + stabilité » : mêmes systèmes (même nombre de planètes, mêmes bornes),
#    planètes intérieures placées au hasard (en log période), refusées si deux voisines ont un rapport < seuil
print("\nmodèle hasard + écart minimal (mêmes systèmes, mêmes bornes) :")
for seuil in (1.0, 1.2, 1.3):
    g = []
    for x in ok:
        k = len(x); lo, hi = x[0], x[-1]; dmin = np.log(seuil)
        if (hi - lo) < (k - 1) * dmin:  # impossible avec ce seuil : on garde le système réel tel quel
            continue
        for _ in range(20):
            while True:
                y = np.sort(np.concatenate([[lo, hi], rng.uniform(lo, hi, k - 2)]))
                if np.min(np.diff(y)) >= dmin: break
            g.append(rapports(y))
    print(f"  seuil {seuil}: <r> = {np.concatenate(g).mean():.4f}  (systèmes utilisables : {len(g)//20})")
# 3) écart minimal observé entre planètes voisines
d = np.concatenate([np.exp(np.diff(x)) for x in ok])
print("\nrapports de périodes voisines : 1er centile", round(np.percentile(d, 1), 3), "| 5e", round(np.percentile(d, 5), 3), "| médiane", round(np.median(d), 3))
