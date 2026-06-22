"""Single source of truth for the model registry.

`MODELS` maps a model string to the module/object exposing `mass_U/mass_D/mass2_W/mass2_Z`, and
`TUNED` maps it to the fundamental Lagrangian-mass parameter basis the Barbieri-Giudice tuning is
differentiated over.  `__init__`, `routes`, `spectrum` and `tuning` all import from here, so a new
model is registered in ONE place rather than five.  (This module is a leaf: the model modules do not
import it, so there is no import cycle.)
"""
from . import mchm5, mchm14, mchm14_1_10, nmchm6, su5so5
from . import assemble

MODELS = {
    "5-5-5": mchm5, "14-14-10": mchm14, "14-1-10": mchm14_1_10,
    "5-5-5-assembled": assemble.model_555, "14-1-10-assembled": assemble.model_14_1_10,
    "14-14-10-assembled": assemble.model_14_14_10, "6-6-6": nmchm6,
    "5-5-su5": su5so5,                  # SU(5)/SO(5), partners in the fundamental 5
    "15-15-su5": su5so5.model_15,       # SU(5)/SO(5), partners in the symmetric 15
}

# fundamental Lagrangian-mass bases differentiated for the tuning (see tuning.py for the conventions)
TUNED_555 = ['mU', 'mUt', 'mYu', 'mSYu', 'mD', 'mDt', 'mYd', 'mSYd',
             'Delta_uL', 'Delta_uR', 'Delta_dL', 'Delta_dR']
TUNED_14 = ['mQ', 'mU', 'mD', 'mYu', 'mSYu', 'mSYtu', 'Yd', 'Delta_q', 'Delta_u', 'Delta_d']
TUNED_14_1_10 = ['mQ', 'mU', 'mD', 'Yu', 'Yd', 'Delta_q', 'Delta_u', 'Delta_d']

TUNED = {
    '5-5-5': TUNED_555, '14-14-10': TUNED_14, '14-1-10': TUNED_14_1_10,
    '5-5-5-assembled': TUNED_555, '14-1-10-assembled': TUNED_14_1_10,
    '14-14-10-assembled': TUNED_14,
    '6-6-6': TUNED_555,            # NMCHM6 shares the 5-5-5 Lagrangian-mass basis (ts=0)
    '5-5-su5': TUNED_555,
    '15-15-su5': TUNED_555,
}
