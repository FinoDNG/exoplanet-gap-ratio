# Exoplanet orbital spacings and the gap-ratio statistic

Data and scripts for the Research Note *"Exoplanet orbital spacings fall in the level-repulsion class of the Riemann zeta zeros, but not on their law"* by David Niedbala Giraudin (ORCID [0009-0009-1526-1178](https://orcid.org/0009-0009-1526-1178)).

Every value in Table 1 of the Note is reproduced by the scripts below.

## Requirements

Python 3 with `numpy`, `scipy` and `mpmath`.
The planet sample comes from the Open Exoplanet Catalogue (Rein 2012), git commit `e03d274e` (2 November 2026):

```
git clone https://github.com/OpenExoplanetCatalogue/open_exoplanet_catalogue.git
cd open_exoplanet_catalogue && git checkout e03d274e && cd ..
```

## Files

| File | Purpose |
|---|---|
| `extraire.py` | Extracts confirmed planets with measured periods, grouped by host, from the catalogue; writes `systemes.json` |
| `systemes.json` | The extracted system list used in the Note |
| `zeros.py` | Computes the first 5,598 zeros of the Riemann zeta function (0 < γ < 6,000) with the Riemann–Siegel formula, checked against mpmath and the Riemann–von Mangoldt count; writes `zeros.npy` |
| `analyse.py` | Mean gap ratio ⟨r⟩ for planets, subsamples, primes, zeta zeros, GUE and Poisson references |
| `robustesse.py` | Removal of the two suspect systems, bootstrap over systems, threshold control models (minimum period ratio 1.0, 1.2, 1.3) |
| `controle_lognormal.py` | Log-normal minimum Hill-spacing controls (Dietrich et al. 2024; fitted to observed minima), mass test, full distribution of r |
| `distribution_r.py` | Share of ratios above 0.8, with bootstrap interval |
| `resultats.json`, `resultats_complement.json` | Outputs of the runs used in the Note |
| `LISEZMOI.txt` | The same instructions in French |

## Running

```
python3 extraire.py open_exoplanet_catalogue
python3 zeros.py
python3 analyse.py open_exoplanet_catalogue
python3 robustesse.py
python3 controle_lognormal.py open_exoplanet_catalogue
python3 distribution_r.py
```

Random draws use fixed seeds, so the outputs are identical from one run to the next.

## Acknowledgments

Computations were carried out with technical assistance from Claude (Anthropic).
