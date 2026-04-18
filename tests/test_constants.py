def test_diseases_mirror_docs():
    from constants.diseases import SUPPORT_DISEASES
    assert len(SUPPORT_DISEASES) == 28
    assert "Aortic regurgitation" in SUPPORT_DISEASES
    assert "Pericardial effusion" in SUPPORT_DISEASES
    # no duplicates
    assert len(set(SUPPORT_DISEASES)) == len(SUPPORT_DISEASES)


def test_measurements_mirror_docs():
    from constants.measurements import SUPPORT_MEASUREMENTS
    assert len(SUPPORT_MEASUREMENTS) == 22
    assert "LV ejection fraction" in SUPPORT_MEASUREMENTS
    assert "Peak LVOT velocity" in SUPPORT_MEASUREMENTS
    assert len(set(SUPPORT_MEASUREMENTS)) == len(SUPPORT_MEASUREMENTS)


def test_report_sections():
    from constants.report_sections import REPORT_SECTIONS
    assert REPORT_SECTIONS == [
        "Aortic Valve",
        "Atria",
        "Great Vessels",
        "Left Ventricle",
        "Mitral Valve",
        "Pericardium Pleural",
        "Pulmonic Valve",
        "Right Ventricle",
        "Tricuspid Valve",
        "Summary",
    ]


def test_view_labels():
    from constants.view_labels import VIEW_LABELS, view_group, is_doppler
    assert len(VIEW_LABELS) == 38
    assert "Apical 4C 2D" in VIEW_LABELS
    assert view_group("Apical 4C 2D") == "a4c"
    assert view_group("Parasternal Long Axis 2D") == "plax"
    from constants.view_labels import view_coarse_group
    assert view_coarse_group("Parasternal Mitral Valve Short Axis") == "psax"
    assert view_group("Subxiphoid IVC 2D") == "ivc"
    assert view_group("Suprasternal Notch") == "ssn"
    assert view_group("Apical 2C 2D") == "a2c"
    assert is_doppler("Parasternal Short Axis Tricuspid Regurgitation CW") is True
    assert is_doppler("Apical 4C 2D") is False


def test_prompts_contain_templates():
    from constants.prompts import REPORT_PROMPT, MEASUREMENT_PROMPT, DISEASE_PROMPT, VQA_PROMPT
    assert "echo report" in REPORT_PROMPT.system.lower() or "echo" in REPORT_PROMPT.system.lower()
    assert "<measure>" in MEASUREMENT_PROMPT.query_template
    assert "<disease>" in DISEASE_PROMPT.query_template
    assert VQA_PROMPT.query_template == "{question}"


def test_presets_subset_of_supported():
    from constants.diseases import SUPPORT_DISEASES
    from constants.measurements import SUPPORT_MEASUREMENTS
    from constants.presets import (
        MEASUREMENT_PRESETS, DISEASE_PRESETS, VQA_EXAMPLES,
    )

    for preset in MEASUREMENT_PRESETS.values():
        for item in preset:
            assert item in SUPPORT_MEASUREMENTS, f"{item} not in SUPPORT_MEASUREMENTS"

    for preset in DISEASE_PRESETS.values():
        for item in preset:
            assert item in SUPPORT_DISEASES, f"{item} not in SUPPORT_DISEASES"

    assert isinstance(VQA_EXAMPLES, list)
    for q in VQA_EXAMPLES:
        assert isinstance(q, str) and 5 <= len(q) <= 160
