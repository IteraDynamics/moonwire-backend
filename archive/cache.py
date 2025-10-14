class SignalCache:
    def __init__(self):
        self._store = {}

    def get_signal(self, key):
        values = self._store.get(key, [])
        return values[-1] if values else None  # Return only the latest signal

    def set_signal(self, key, value):
        if key not in self._store:
            self._store[key] = []
        self._store[key].append(value)

    def clear(self):
        self._store.clear()

    def keys(self):
        return self._store.keys()