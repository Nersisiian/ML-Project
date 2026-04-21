def test_always_passes():
    assert True
    print("✅ CI test passed!")
def test_project_structure():
    import os
    assert os.path.exists("app")
    assert os.path.exists("ml")
    print("✅ Structure OK!")
