'''Shared utility functions'''

from pathlib import Path
import numpy as np

def get_project_root() -> Path:
    '''Get the root dir of the project'''
    return Path(__file__).parent.parent