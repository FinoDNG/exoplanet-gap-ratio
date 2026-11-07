"""Extrait les systèmes de planètes confirmées (période connue) du catalogue Open Exoplanet Catalogue."""
import glob, json, xml.etree.ElementTree as ET
from collections import Counter

def val(el, tag):
    x = el.find(tag)
    if x is None or x.text is None: return None
    try: return float(x.text)
    except ValueError: return None

systemes = []
for f in sorted(glob.glob("open_exoplanet_catalogue/systems/*.xml")):
    root = ET.parse(f).getroot()
    sysname = root.findtext("name")
    for parent in root.iter():
        pl = [c for c in parent if c.tag == "planet"]
        if not pl: continue
        planetes = []
        for p in pl:
            listes = [l.text for l in p.findall("list")]
            if "Confirmed planets" not in listes: continue
            per = val(p, "period")
            if per is None or per <= 0: continue
            planetes.append({"nom": p.findtext("name"), "periode": per, "a": val(p, "semimajoraxis"),
                             "rayon": val(p, "radius"), "masse": val(p, "mass"),
                             "methode": p.findtext("discoverymethod"), "transit": p.findtext("istransiting")})
        if not planetes: continue
        planetes.sort(key=lambda d: d["periode"])
        hote = parent.findtext("name") if parent.tag != "system" else sysname
        systemes.append({"fichier": f.split("/")[-1], "systeme": sysname, "hote": hote, "type_hote": parent.tag,
                         "planetes": planetes})

json.dump(systemes, open("systemes.json", "w"), indent=0, ensure_ascii=False)
mult = Counter(len(s["planetes"]) for s in systemes)
print("groupes de planètes (par hôte) :", len(systemes), "| planètes :", sum(len(s["planetes"]) for s in systemes))
print("multiplicité :", dict(sorted(mult.items())))
print("systèmes ≥ 3 planètes :", sum(v for k, v in mult.items() if k >= 3))
# planètes de même période dans un système (doublons possibles)
dup = [(s["systeme"], [p["periode"] for p in s["planetes"]]) for s in systemes
       if len({round(p["periode"], 4) for p in s["planetes"]}) < len(s["planetes"])]
print("systèmes avec périodes identiques :", len(dup), dup[:5])
sol = [s for s in systemes if "Sun" in s["systeme"] or "Solar" in s["systeme"]]
print("Système solaire présent ?", [(s["systeme"], len(s["planetes"])) for s in sol])
