"""
Test if we can import the AFP modules.
"""

import sys
import os

# Add the current directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath('.'))

try:
    print("Trying to import message module...")
    from moya.communication.afp.message import AFPMessage, ContentType
    print("Successfully imported message module")
    
    print("\nTrying to import exceptions module...")
    from moya.communication.afp.exceptions import AFPError, AFPMessageError
    print("Successfully imported exceptions module")
    
    print("\nTrying to import subscription module...")
    from moya.communication.afp.subscription import AFPSubscription
    print("Successfully imported subscription module")
    
    print("\nAll imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    
    # Print the Python path
    print("\nPython path:")
    for path in sys.path:
        print(f"  {path}")
    
    # Check if the files exist
    print("\nChecking if files exist:")
    files = [
        "moya/communication/__init__.py",
        "moya/communication/afp/__init__.py",
        "moya/communication/afp/message.py",
        "moya/communication/afp/exceptions.py",
        "moya/communication/afp/subscription.py"
    ]
    for file in files:
        exists = os.path.exists(file)
        print(f"  {file}: {'exists' if exists else 'does not exist'}")
except Exception as e:
    print(f"Unexpected error: {e}") 