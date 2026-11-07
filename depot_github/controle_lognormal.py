"""Complément de la note : (1) modèle témoin avec distance minimale LOG-NORMALE en rayons de Hill mutuels
(Dietrich, Malhotra & Apai 2024, AJ 167, 46 : Delta_min ~ LogNormal(mu = 1,97, sigma = 0,40), médiane 7,2) ;
(2) distribution complète de r, en particulier la queue des petits rapports (signature de la répulsion).

Espacement de Hill d'une paire voisine (périodes P1 < P2, masses m1, m2, étoile M) :
  x = (P2/P1)^(2/3) = a2/a1,  Delta = 2 (x - 1)/(x + 1) / [ (m1 + m2)/(3 M) ]^(1/3).
Masses : masse du catalogue si présente (m sin i pour les vitesses radiales) ; sinon déduite du rayon
(Weiss & Marcy 2014 pour R < 4 R_terre ; Chen & Kipping 2017, branche neptunienne R = 0,80 M^0,59 au-delà,
plafonnée à 1 masse de Jupiter) ; sinon 10 masses terrestres. Étoile : masse du catalogue, sinon 1 masse solaire.
Usage : python3 controle_lognormal.py ../open_exoplanet_catalogue   (dossier du catalogue, commit e03d274e)"""
import sys, glob, json, xml.etree.ElementTree as ET
import numpy as np, os
FACT = float(os.environ.get("FACTEUR_MASSE", "1"))   # sensibilité : facteur appliqué aux masses déduites des rayons
RAPIDE = os.environ.get("RAPIDE") == "1"

rng = np.random.default_rng(20261106)
MJ_ME = 317.83          # masse de Jupiter en masses terrestres
RJ_RE = 11.209          # rayon de Jupiter en rayons terrestres
MS_ME = 332946.0        # masse du Soleil en masses terrestres
MU, SIG = 1.97, 0.40

# ---------------------------------------------------------------- masses des étoiles hôtes
cat = sys.argv[1] if len(sys.argv) > 1 else "../open_exoplanet_catalogue"
def fval(el, tag):
    x = el.find(tag)
    try: return float(x.text) if x is not None and x.text else None
    except ValueError: return None
mstar = {}
for f in sorted(glob.glob(cat + "/systems/*.xml")):
    root = ET.parse(f).getroot(); sysname = root.findtext("name")
    for parent in root.iter():
        if any(c.tag == "planet" for c in parent):
            hote = parent.findtext("name") if parent.tag != "system" else sysname
            mstar[(f.split("/")[-1], hote)] = fval(parent, "mass") if parent.tag == "star" else None

def masse_terre(p):
    if p["masse"] is not None and p["masse"] > 0: return p["masse"] * MJ_ME, "catalogue"
    if p["rayon"] is not None and p["rayon"] > 0:
        R = p["rayon"] * RJ_RE
        if R < 1.5: M = (2.43 + 3.39 * R) / 5.51 * R ** 3         # densité de Weiss & Marcy, en masses terrestres
        elif R < 4: M = 2.69 * R ** 0.93
        else: M = (R / 0.80) ** (1 / 0.59)
        return min(M, MJ_ME) * FACT, "rayon"
    return 10.0, "défaut"

def rapports(x):
    s = np.diff(np.sort(np.asarray(x, float))); a, b = s[:-1], s[1:]
    return np.minimum(a, b) / np.maximum(a, b)

def hill(lp, m, M):
    x = np.exp(np.diff(lp) * 2 / 3)
    return 2 * (x - 1) / (x + 1) / ((m[:-1] + m[1:]) / (3 * M)) ** (1 / 3)

S = [s for s in json.load(open("systemes.json")) if len(s["planetes"]) >= 3]
S = [s for s in S if np.min(np.diff(np.log([p["periode"] for p in s["planetes"]]))) >= np.log(1.1)]   # mêmes 267 systèmes
print("systèmes :", len(S))
origine = {"catalogue": 0, "rayon": 0, "défaut": 0}; etoile_def = 0
data = []
for s in S:
    lp = np.log([p["periode"] for p in s["planetes"]])
    ms = []
    for p in s["planetes"]:
        m, o = masse_terre(p); ms.append(m); origine[o] += 1
    M = mstar.get((s["fichier"], s["hote"]))
    if not M: M = 1.0; etoile_def += 1
    data.append((lp, np.array(ms), M * MS_ME))
print("masses des planètes :", origine, "| étoiles sans masse (1 masse solaire prise) :", etoile_def)

obs = [rapports(lp) for lp, m, M in data]
r_obs = np.concatenate(obs)
dmin_obs = np.array([hill(lp, m, M).min() for lp, m, M in data])
print(f"observé : <r> = {r_obs.mean():.4f} ; Delta minimal par système : médiane {np.median(dmin_obs):.2f}, "
      f"moyenne de ln {np.log(dmin_obs).mean():.2f}, écart-type de ln {np.log(dmin_obs).std():.2f} "
      f"(Dietrich et al. pour des systèmes stables simulés : 1,97 et 0,40)")

# ---------------------------------------------------------------- modèle témoin log-normal
def tirage(lp, m, M, dmin, essais=20000):
    k = len(lp); lo, hi = lp[0], lp[-1]
    for _ in range(essais):
        y = np.sort(np.concatenate([[lo, hi], rng.uniform(lo, hi, k - 2)]))
        if np.all(np.diff(y) > 0) and hill(y, m, M).min() >= dmin: return y
    return None

def modele(nom, gen_dmin, ntir=20):
    g, abandon = [], 0
    for lp, m, M in data:
        gs = []
        for _ in range(ntir):
            y = None
            for _ in range(50):                       # on retire Delta_min si le tirage est impossible
                y = tirage(lp, m, M, gen_dmin(), essais=2000)
                if y is not None: break
            if y is None: abandon += 1; continue
            gs.append(rapports(y))
        if gs: g.append(np.concatenate(gs))
    r = np.concatenate(g)
    print(f"  {nom:44s} <r> = {r.mean():.4f}   r < 0,1 : {np.mean(r < 0.1) * 100:4.1f} %   (tirages abandonnés : {abandon})")
    return r

print("\nmodèles témoins (mêmes systèmes, mêmes bornes, masses réelles, 20 tirages par système) :")
if RAPIDE:
    modele(f"Dietrich log-normale, masses déduites x {FACT}", lambda: rng.lognormal(MU, SIG)); sys.exit(0)
r0 = modele("aucune contrainte", lambda: 0.0)
rL = modele("Delta_min log-normale (Dietrich et al. 2024)", lambda: rng.lognormal(MU, SIG))
rF = modele("Delta_min fixe = médiane 7,17", lambda: 7.17)
mu_o, sig_o = float(np.log(dmin_obs).mean()), float(np.log(dmin_obs).std())
rO = modele(f"Delta_min log-normale AJUSTÉE aux observés ({mu_o:.2f} ; {sig_o:.2f})", lambda: rng.lognormal(mu_o, sig_o))

# ---------------------------------------------------------------- distribution complète de r
Zs = np.load("zeros.npy") if len(sys.argv) < 3 else np.load(sys.argv[2])
rz = rapports(Zs)
rp = rapports(np.cumsum(rng.exponential(size=1_000_000)))
rg = []
for _ in range(300):
    A = rng.normal(size=(200, 200)) + 1j * rng.normal(size=(200, 200)); ev = np.linalg.eigvalsh((A + A.conj().T) / 2)
    rg.append(rapports(ev[50:150]))
rg = np.concatenate(rg)

def boot_frac(groupes, f, n=4000):
    k = len(groupes); v = [f(np.concatenate([groupes[i] for i in rng.integers(0, k, k)])) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])
bins = np.linspace(0, 1, 6)
print(f"\nzéros utilisés : {len(Zs)} (0 < t < {Zs.max():.0f})")
print("répartition de r par tranches de 0,2 (%), et part des petits rapports :")
lo, hi = boot_frac(obs, lambda r: np.mean(r < 0.1) * 100)
lo2, hi2 = boot_frac(obs, lambda r: np.mean(r < 0.2) * 100)
for nom, r in (("planètes", r_obs), ("témoin log-normal", rL), ("témoin ajusté", rO), ("témoin sans contrainte", r0), ("zéros de zêta", rz), ("GUE", rg), ("Poisson", rp)):
    h = np.histogram(r, bins)[0] / len(r) * 100
    print(f"  {nom:24s}", "  ".join(f"{x:5.1f}" for x in h), f"| r<0,1 : {np.mean(r < 0.1) * 100:4.1f}  r<0,2 : {np.mean(r < 0.2) * 100:4.1f}")
print(f"  intervalle à 95 % (rééchantillonnage des systèmes) pour les planètes : r<0,1 [{lo:.1f} ; {hi:.1f}], r<0,2 [{lo2:.1f} ; {hi2:.1f}]")
json.dump({"r_obs": float(r_obs.mean()), "r_libre": float(r0.mean()), "r_lognormal": float(rL.mean()), "r_fixe_7.17": float(rF.mean()), "r_ajuste": float(rO.mean()), "mu_sig_obs": [mu_o, sig_o],
           "petits_r<0.1": {"planetes": float(np.mean(r_obs < 0.1)), "lognormal": float(np.mean(rL < 0.1)), "zeros": float(np.mean(rz < 0.1)),
                            "GUE": float(np.mean(rg < 0.1)), "Poisson": float(np.mean(rp < 0.1))}},
          open("resultats_complement.json", "w"), indent=1)
