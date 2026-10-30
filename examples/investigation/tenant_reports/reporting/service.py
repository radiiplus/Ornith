from .auth import allowed
from .render import render


class ReportService:
    def __init__(self, store, cache):
        self.store = store
        self.cache = cache

    def report(self, user, tenant):
        if not allowed(user, tenant):
            raise PermissionError('tenant access denied')
        key = user['id']
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = render(tenant, self.store.totals(tenant))
        self.cache.put(key, result)
        return result
