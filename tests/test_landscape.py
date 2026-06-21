"""The coset-landscape enumerator reproduces a slice of the Chala-Fonseca (arXiv:2309.10635)
classification from pyCHM's own Coset abstraction: it scans G/H families, branches each coset under
a custodial SO(4), and flags those with a (2,2) Higgs."""
from pychm.groups import landscape


def test_scan_reproduces_pyCHM_models():
    rows = {r['coset']: r for r in landscape.scan(max_ngb=14)}
    # the pyCHM models, with their known pNGB content under custodial SO(4)
    assert rows['SO(5)/SO(4)']['n_pngb'] == 4 and rows['SO(5)/SO(4)']['content'] == '(2,2)'        # MCHM
    assert rows['SO(6)/SO(5)']['n_pngb'] == 5 and rows['SO(6)/SO(5)']['content'] == '(1,1) + (2,2)'  # NMCHM
    assert rows['SU(5)/SO(5)']['n_pngb'] == 14 and '(3,3)' in rows['SU(5)/SO(5)']['content']        # littlest Higgs
    assert all(r['has_higgs'] for r in rows.values())            # every listed coset has the (2,2) Higgs


def test_scan_discovers_larger_cosets():
    rows = {r['coset']: r for r in landscape.scan(max_ngb=14)}
    # the SO(N)/SO(N-1) tower: N-1 pNGBs = a (2,2) + (N-4) singlets
    assert rows['SO(7)/SO(6)']['content'] == '2x(1,1) + (2,2)'
    assert rows['SO(8)/SO(7)']['content'] == '3x(1,1) + (2,2)'


def test_higgs_criterion_is_the_2_2():
    # the enumerator's filter is exactly "the coset contains a (2,2) of custodial SO(4)"
    rows = landscape.scan(max_ngb=14)
    for r in rows:
        assert r['has_higgs'] == ('(2,2)' in r['content'])
