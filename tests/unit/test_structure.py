"""
Test the basic structure of the AFP implementation.
"""

import os
import unittest
import importlib.util
import sys

class TestAFPStructure(unittest.TestCase):
    """Test the structure of the AFP module."""
    
    def setUp(self):
        """Set up the test case by determining the project root directory."""
        # Find the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(current_dir, '../..'))
    
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
            full_path = os.path.join(self.project_root, module_path)
            exists = os.path.exists(full_path)
            status = '✓' if exists else '✗'
            print(f"  {module_path}: {status}")
            self.assertTrue(exists, f"Module {module_path} does not exist")
    
    def test_version(self):
        """Test that the AFP module has a version number."""
        module_path = "moya/communication/afp/__init__.py"
        full_path = os.path.join(self.project_root, module_path)
        
        print(f"\nTesting version in {module_path}")
        if not os.path.exists(full_path):
            self.fail(f"File {module_path} does not exist")
            
        with open(full_path, 'r') as f:
            content = f.read()
            print(f"  File content:\n{content.strip()}")
            self.assertIn("__version__", content, "AFP module missing __version__ attribute")

if __name__ == "__main__":
    print("Running AFP Structure Tests...")
    unittest.main(verbosity=2) 