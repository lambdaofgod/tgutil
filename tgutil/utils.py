from typing import Dict, Any
import hashlib
import json
from fastapi.encoders import jsonable_encoder
import string


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(jsonable_encoder(dictionary), sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def _strip_punctuation(s):
    return s.translate(str.maketrans("", "", string.punctuation))
