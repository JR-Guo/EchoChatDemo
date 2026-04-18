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
