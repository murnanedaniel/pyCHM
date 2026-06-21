"""A coset-landscape enumerator built on the engine.

For each `(G, H)` pair in the families the engine supports, build the coset, compute the pNGB
content by branching the coset under a custodial `SO(4) = SU(2)_L x SU(2)_R` in `H`, and flag those
containing a `(2,2)` Higgs doublet -- the criterion of Chala & Fonseca, "The Landscape of Composite
Higgs Models" (arXiv:2309.10635), whose 642-model classification this reproduces a slice of.

Covered families (clean custodial SO(4) on indices 0..3 of the defining rep): SO(N)/SO(N-1) and
SU(N)/SO(N).  SU(N)/USp(N) (e.g. SU(4)/Sp(4)) is the third UV-completable family; its custodial
SO(4) sits inside USp(N) and is reached via the SU(4)/Sp(4) ~= SO(6)/SO(5) isomorphism (the NMCHM
row).  Each model pyCHM implements appears in the scan with its correct pNGB content.
"""
import numpy as np

from . import coset as _coset, decompose as _D


def _so4_generators(n):
    """The six custodial SO(4) generators (Hermitian, indices 0..3) in the n-dim defining rep."""
    M = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            T = np.zeros((n, n), dtype=complex)
            T[mu, nu] = -1j
            T[nu, mu] = 1j
            M[(mu, nu)] = T
    return M


def coset_so4_content(cos):
    """SO(4) = SU(2)_L x SU(2)_R content {(jL,jR): mult} of the coset, from the adjoint action of
    the custodial SO(4) (indices 0..3) on the broken generators."""
    so4 = _so4_generators(cos.dim)
    B = np.array([g.ravel() for g in cos.broken])
    coeffs = lambda Mx: np.linalg.lstsq(B.T, Mx.ravel(), rcond=None)[0]   # [T,X]=i*(real) -> complex
    adj = {}
    for key, Tg in so4.items():
        A = np.zeros((len(cos.broken), len(cos.broken)), dtype=complex)
        for a, Xa in enumerate(cos.broken):
            A[:, a] = coeffs(Tg @ Xa - Xa @ Tg)
        adj[key] = A
    table = {}
    for (jL, jR), cols in _D.so4_decompose_gen(adj):
        m = round(cols.shape[1] / ((2 * jL + 1) * (2 * jR + 1)))
        table[(jL, jR)] = table.get((jL, jR), 0) + m
    return table


def _content_str(content):
    name = {(0.0, 0.0): '(1,1)', (0.5, 0.5): '(2,2)', (1.0, 0.0): '(3,1)', (0.0, 1.0): '(1,3)',
            (1.0, 1.0): '(3,3)', (1.5, 0.5): '(4,2)', (0.5, 1.5): '(2,4)'}
    parts = []
    for k in sorted(content, key=lambda t: (t[0] + t[1], t)):
        lbl = name.get(k, f'({2*k[0]+1:g},{2*k[1]+1:g})')
        parts.append(lbl if content[k] == 1 else f'{content[k]}x{lbl}')
    return ' + '.join(parts)


def scan(max_ngb=14, n_max=8):
    """Enumerate SO(N)/SO(N-1) and SU(N)/SO(N) cosets up to `max_ngb` pNGBs.  Returns a list of
    dicts: {coset, n_pngb, content, has_higgs (a (2,2)), symmetric (always True here -- both
    families are symmetric spaces)}."""
    rows = []
    for N in range(5, n_max + 1):
        for name, builder in ((f"SO({N})/SO({N-1})", lambda N=N: _coset.so_coset(N)),
                              (f"SU({N})/SO({N})", lambda N=N: _coset.su_so_coset(N))):
            cos = builder()
            if cos.n_pngb > max_ngb:
                continue
            content = coset_so4_content(cos)
            rows.append(dict(coset=name, n_pngb=cos.n_pngb, content=_content_str(content),
                             has_higgs=(0.5, 0.5) in content, symmetric=True))
    rows.sort(key=lambda r: (r['n_pngb'], r['coset']))
    return rows


def print_scan(max_ngb=14, n_max=8):
    print(f"{'coset':<14}{'#pNGB':>6}  {'has (2,2)?':<11}content")
    for r in scan(max_ngb, n_max):
        print(f"{r['coset']:<14}{r['n_pngb']:>6}  {'YES' if r['has_higgs'] else 'no':<11}{r['content']}")
