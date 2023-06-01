from __future__ import annotations

import json
import os
import pickle
import hashlib


def _make_hash(o: dict) -> str:
    """Make a hash of a dictionary"""
    j = json.dumps(o, sort_keys=True)
    return hashlib.md5(j.encode('utf-8')).hexdigest()


class CacheManager:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Read cache index if it exists
        if os.path.exists(os.path.join(cache_dir, 'index.json')):
            with open(os.path.join(cache_dir, 'index.json')) as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def repair(self, key: dict):
        if key in self:
            print(f'Key {key} already exists in cache')
        else:
            if os.path.isfile(os.path.join(self.cache_dir, self.get_file_path(key))):
                self[key] = key
                print(f'Key {key} added to cache')
            else:
                print(f'Key {key} not found in cache directory')

    def load(self, key: dict):
        if key not in self:
            raise KeyError(f'Key {key} not found in cache')

        with open(os.path.join(self.cache_dir, self.get_file_path(key)), 'rb') as handle:
            data = pickle.load(handle)
        return data

    def save(self, key: dict, value, force=False):
        if key in self and not force:
            raise KeyError(f'Key {key} already exists in cache')
        self[key] = key
        with open(os.path.join(self.cache_dir, self.get_file_path(key)), 'wb') as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def delete(self, key: dict):
        if key not in self:
            raise KeyError(f'Key {key} not found in cache')
        del self[key]
        os.remove(os.path.join(self.cache_dir, self.get_file_path(key)))

    def get_file_path(self, key: dict, file_ending: str = 'pickle') -> str:
        return os.path.join(self.cache_dir, f'{_make_hash(key)}.{file_ending}')

    def clear(self):
        self.cache = {}
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def __contains__(self, key: str | dict) -> bool:
        if isinstance(key, dict):
            key = _make_hash(key)
        return key in self.cache

    def __getitem__(self, key: str | dict):
        if isinstance(key, dict):
            key = _make_hash(key)
        return self.cache[key]

    def __setitem__(self, key: str | dict, value):
        if isinstance(key, dict):
            key = _make_hash(key)
        self.cache[key] = value
        with open(os.path.join(self.cache_dir, 'index.json'), 'w') as f:
            json.dump(self.cache, f)

    def __delitem__(self, key: str | dict):
        if isinstance(key, dict):
            key = _make_hash(key)
        del self.cache[key]
        with open(os.path.join(self.cache_dir, 'index.json'), 'w') as f:
            json.dump(self.cache, f)
