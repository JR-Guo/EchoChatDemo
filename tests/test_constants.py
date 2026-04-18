def test_diseases_mirror_docs():
    from constants.diseases import SUPPORT_DISEASES
    assert len(SUPPORT_DISEASES) == 28
    assert "Aortic regurgitation" in SUPPORT_DISEASES
    assert "Pericardial effusion" in SUPPORT_DISEASES
    # no duplicates
    assert len(set(SUPPORT_DISEASES)) == len(SUPPORT_DISEASES)
