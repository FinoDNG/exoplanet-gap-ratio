"""Zéros de zêta sur la droite critique jusqu'à T = 6000, par la formule de Riemann–Siegel,
puis contrôle : (1) comparaison aux 200 premiers zéros de mpmath, (2) comptage de Riemann–von Mangoldt."""
import numpy as np, mpmath
from scipy.optimize import brentq

def theta(t):
    return t/2*np.log(t/(2*np.pi)) - t/2 - np.pi/8 + 1/(48*t) + 7/(5760*t**3)

def Z(t):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    a = np.sqrt(t/(2*np.pi)); N = np.floor(a).astype(int); Nmax = N.max()
    n = np.arange(1, Nmax+1)[:, None]
    terms = np.where(n <= N[None, :], np.cos(theta(t)[None, :] - t[None, :]*np.log(n))/np.sqrt(n), 0.0)
    s = 2*terms.sum(0)
    p = a - N
    C0 = np.cos(2*np.pi*(p**2 - p - 1/16))/np.cos(2*np.pi*p)
    return s + (-1)**(N-1) * (2*np.pi/t)**0.25 * C0

T = 6000.0
grid = np.arange(10.0, T, 0.005)
vals = np.concatenate([Z(grid[i:i+20000]) for i in range(0, len(grid), 20000)])
idx = np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))
zeros = np.array([brentq(lambda x: Z(x)[0], grid[i], grid[i+1], xtol=1e-10) for i in idx])
ref = np.array([float(mpmath.zetazero(n).imag) for n in range(1, 201)])
print("zéros trouvés jusqu'à", T, ":", len(zeros))
print("écart max avec mpmath (200 premiers) :", np.max(np.abs(zeros[:200] - ref)))
# Riemann–von Mangoldt : N(T) = theta(T)/pi + 1 + S(T), |S(T)| petit
for t in (1000, 2000, 4000, 5999):
    print(t, "comptés", int(np.sum(zeros < t)), " prévu (sans S)", round(theta(t)/np.pi + 1, 2))
np.save("zeros.npy", zeros)
