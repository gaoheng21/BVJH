"""Repository resources resolved without changing the process working directory."""
import os
from pathlib import Path

REPOSITORY = Path(os.environ.get('INTERACTIONPRISM_ROOT', Path(__file__).resolve().parent.parent)).resolve()

def resource_path(*parts):
    path = Path(*parts).expanduser()
    return path if path.is_absolute() else REPOSITORY / path

def provenance_key(path):
    """Keep archived E8 hash keys compatible with the byte-identical provenance copy."""
    if Path(path).name == 'primitives_provenance.txt':
        return 'IP_E8R1_20261111_e4_primitives.py'
    return Path(path).name
