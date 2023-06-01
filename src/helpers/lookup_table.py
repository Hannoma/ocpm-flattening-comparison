import json
import hashlib


def _calculate_hash(d):
    """Make a hash of a _dictionary"""
    j = json.dumps(d, sort_keys=True)
    return int.from_bytes(hashlib.md5(j.encode('utf-8')).digest(), byteorder='little')


class LookupTable(dict):

    def __init__(self, mapper):
        super().__init__(self, name=f'Lookup table with {len(mapper)} entries')
        self._dict = mapper
        self.hash = _calculate_hash(mapper)

    def __hash__(self):
        return self.hash

    def __contains__(self, item):
        return item in self._dict

    def __getitem__(self, item):
        return self._dict[item]

    def __str__(self):
        return f'Lookup table with {len(self._dict)} entries'

    def __repr__(self):
        return str(self)
