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
