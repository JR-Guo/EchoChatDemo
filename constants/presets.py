"""Preset bundles for Measurement and Disease tasks, plus VQA example questions.

Presets let users run a grouped selection in one click. Keys must remain
subsets of SUPPORT_MEASUREMENTS / SUPPORT_DISEASES, verified in the test suite.
"""

MEASUREMENT_PRESETS: dict[str, list[str]] = {
    "Left Ventricle": [
        "LV ejection fraction",
        "Left-ventricular posterior-wall thickness in diastole",
        "Interventricular-septal thickness in diastole",
    ],
    "Left Atrium": [
        "Left-atrial antero-posterior dimension",
        "Left-atrial volume at end-systole",
    ],
    "Aortic Valve": [
        "Aortic-root diameter",
        "Peak systolic velocity across the aortic valve",
        "Peak systolic pressure gradient across the aortic valve",
    ],
    "Mitral Inflow": [
        "Mitral inflow A-wave peak velocity",
        "Mitral inflow E-wave peak velocity",
        "Peak transmitral velocity",
        "Peak transmitral pressure gradient",
    ],
    "Diastolic Function": [
        "Septal mitral-annulus early-diastolic tissue velocity (E')",
        "Avg E/E'",
    ],
    "Right Heart": [
        "Tricuspid-annular-plane systolic excursion",
        "Right-ventricular lateral-annulus systolic tissue velocity (S')",
        "Peak velocity of TR jet",
        "Peak pressure gradient of TR jet",
    ],
    "Pulmonary": [
        "Peak velocity across pulmonary valve",
        "Peak pressure gradient across pulmonary valve",
    ],
    "LVOT": [
        "Peak LVOT velocity",
        "Peak pressure gradient across LVOT",
    ],
}

DISEASE_PRESETS: dict[str, list[str]] = {
    "Aortic Valve": [
        "Aortic regurgitation",
        "Aortic stenosis",
        "Bicuspid aortic valve",
        "Aortic root dilation",
    ],
    "Mitral Valve": [
        "Mitral regurgitation",
        "Mitral stenosis",
    ],
    "Right Heart": [
        "Tricuspid Regurgitation",
        "Pulmonary regurgitation",
        "Pulmonary artery dilation",
        "Pulmonary hypertension",
        "Right-atrial Dilation",
        "Right-ventricular Dilation",
        "Right-ventricular systolic dysfunction",
    ],
    "Left Heart": [
        "Left-atrial Dilation",
        "Left-ventricular Dilation",
        "Left-ventricular apical aneurysm",
        "Left-ventricular diastolic dysfunction",
        "Left-ventricular systolic dysfunction",
    ],
    "Septal Defects": [
        "Atrial septal defect",
        "Ventricular septal defect",
    ],
    "Post-op / Devices": [
        "Pacemaker in situ",
        "Status post aortic-valve replacement",
        "Status post mitral-valve replacement",
        "Mechanical prosthetic valve (post valve replacement)",
    ],
    "Other": [
        "Inferior vena cava dilation",
        "Hypertrophic cardiomyopathy",
        "Segmental wall-motion abnormality",
        "Pericardial effusion",
    ],
}

VQA_EXAMPLES: list[str] = [
    "What view is shown in this clip?",
    "Is there any obvious pericardial effusion?",
    "Describe the left ventricular wall motion.",
    "Does the mitral valve appear thickened?",
    "Is there evidence of left ventricular hypertrophy?",
    "Estimate the left ventricular ejection fraction.",
    "Are the aortic valve leaflets normally mobile?",
    "Is there color Doppler evidence of tricuspid regurgitation?",
]
