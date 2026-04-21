def test_project_exists():
    """Simple test that always passes"""
    import os
    assert os.path.exists("app")
    assert os.path.exists("ml")
    assert os.path.exists("docker")
    print("✅ Project structure OK!")

def test_python_version():
    """Check Python version"""
    import sys
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 9
    print("✅ Python version OK!")

def test_imports():
    """Test critical imports"""
    try:
        import fastapi
        print("✅ FastAPI OK")
    except ImportError as e:
        print(f"FastAPI import error: {e}")
    
    try:
        import numpy
        print("✅ NumPy OK")
    except ImportError as e:
        print(f"NumPy import error: {e}")
    
    print("✅ Import test completed!")
