"""
Test the basic structure of the AFP implementation.
"""

import os
import unittest
import importlib.util

class TestAFPStructure(unittest.TestCase):
    """Test the structure of the AFP module."""
    
    def test_modules_exist(self):
        """Test that all required modules exist."""
        modules = [
            "moya/communication/__init__.py",
            "moya/communication/afp/__init__.py",
            "moya/communication/afp/security/__init__.py",
            "moya/communication/afp/reliability/__init__.py",
            "moya/communication/afp/monitoring/__init__.py",
        ]
        
        print("\nTesting module existence:")
        for module_path in modules:
            exists = os.path.exists(module_path)
            status = '✓' if exists else '✗'
            print(f"  {module_path}: {status}")
            self.assertTrue(exists, f"Module {module_path} does not exist")
    
    def test_version(self):
        """Test that the AFP module has a version number."""
        module_path = "moya/communication/afp/__init__.py"
        
        print(f"\nTesting version in {module_path}")
        if not os.path.exists(module_path):
            self.fail(f"File {module_path} does not exist")
            
        with open(module_path, 'r') as f:
            content = f.read()
            print(f"  File content:\n{content.strip()}")
            self.assertIn("__version__", content, "AFP module missing __version__ attribute")

if __name__ == "__main__":
    print("Running AFP Structure Tests...")
    unittest.main(verbosity=2) 