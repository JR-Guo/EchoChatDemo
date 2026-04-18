"""38 view classes as emitted by EchoView38Classifier on port 8995.

Groups are coarse buckets the UI uses to render the Study Panel badges.
Doppler/spectrum variants are NOT subclassified (per product decision):
the UI rolls them into a single 'Doppler / Spectrum' chip count.
"""

VIEW_LABELS: list[str] = [
    "Apical 4C 2D",
    "Apical 2C 2D",
    "Apical 3C 2D",
    "Parasternal Mitral Valve Short Axis",
    "Parasternal Papillary Muscle Short Axis 2D",
    "Parasternal Apical Short Axis 2D",
    "Apical 5C AV CW",
    "Subxiphoid IVC 2D",
    "Parasternal Long Axis 2D",
    "Parasternal Long Axis of the Pulmonary Artery",
    "Parasternal Short Axis Tricuspid PW",
    "Apical 4C Color",
    "Apical 4C MV Annulus TDI PW",
    "Parasternal Long Axis Color",
    "Apical 4C MV PW",
    "Parasternal Short Axis Tricuspid Color",
    "Parasternal Long Axis M-mode",
    "Apical 4C Right Ventricular Focus",
    "Parasternal Right Ventricular Outflow Tract Color",
    "Parasternal Right Ventricular Outflow Tract RVOT PW",
    "Parasternal Right Ventricular Outflow Tract PA PW",
    "Apical 4C TV Annulus TDI PW",
    "Apical 4C TAPSE",
    "Parasternal Right Ventricular Inflow Tract Color",
    "Parasternal Right Ventricular Inflow Tract 2D",
    "Parasternal Short Axis Tricuspid Regurgitation CW",
    "Subxiphoid IVC M-mode",
    "Suprasternal Notch",
    "Parasternal Right Ventricular Outflow Tract PR CW",
    "A4C LVO",
    "A3C LVO",
    "A4C MCE",
    "A3C MCE",
    "A2C MCE",
    "A2C LVO",
    "A4C MCE Flash",
    "A2C MCE Flash",
    "A3C MCE Flash",
]

_SPECTRUM_TOKENS = ("PW", "CW", "TDI", "M-mode", "Color", "LVO", "MCE")


def is_doppler(label: str) -> bool:
    return any(tok in label for tok in _SPECTRUM_TOKENS)


_GROUP_PREFIXES: list[tuple[str, str]] = [
    ("Apical 4C", "a4c"),
    ("A4C", "a4c"),
    ("Apical 2C", "a2c"),
    ("A2C", "a2c"),
    ("Apical 3C", "a3c"),
    ("A3C", "a3c"),
    ("Apical 5C", "a5c"),
    ("Parasternal Long Axis of the Pulmonary Artery", "pala"),
    ("Parasternal Long Axis", "plax"),
    ("Parasternal Mitral Valve Short Axis", "psax_mv"),
    ("Parasternal Papillary Muscle Short Axis", "psax_pm"),
    ("Parasternal Apical Short Axis", "psax_apex"),
    ("Parasternal Short Axis", "psax"),
    ("Parasternal Right Ventricular Outflow Tract", "rvot"),
    ("Parasternal Right Ventricular Inflow Tract", "rvit"),
    ("Subxiphoid IVC", "ivc"),
    ("Suprasternal Notch", "ssn"),
]


def view_group(label: str) -> str:
    """Map a raw label to a coarse group key.

    Groups named 'psax_*' collapse to 'psax' for simple 'has-PSAX' checks
    via view_coarse_group.
    """
    for prefix, key in _GROUP_PREFIXES:
        if label.startswith(prefix):
            return key
    return "other"


def view_coarse_group(label: str) -> str:
    g = view_group(label)
    if g.startswith("psax"):
        return "psax"
    return g
