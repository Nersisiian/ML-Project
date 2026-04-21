def test_ci_works():
    """Basic CI test that always passes"""
    assert True
    print("✅ CI test passed!")

def test_project_has_app():
    """Check that app directory exists"""
    import os
    assert os.path.exists("app"), "app directory not found"
    assert os.path.exists("ml"), "ml directory not found"
    assert os.path.exists("docker"), "docker directory not found"
    print("✅ Project structure OK!")

def test_python_version():
    """Check Python version"""
    import sys
    assert sys.version_info.major >= 3
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK!")
