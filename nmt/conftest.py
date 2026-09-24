# Make the repository root importable (utils/, model/, data/) when pytest is
# invoked from anywhere.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
