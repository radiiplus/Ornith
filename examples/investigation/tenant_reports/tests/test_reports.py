import unittest

from reporting.cache import ReportCache
from reporting.service import ReportService
from reporting.store import ReportStore


class ReportTests(unittest.TestCase):
    def test_switching_tenants_does_not_reuse_another_report(self):
        service = ReportService(ReportStore({'north': [20, 30], 'south': [7, 8]}), ReportCache())
        user = {'id': 'analyst', 'tenants': {'north', 'south'}}
        self.assertEqual(service.report(user, 'north'), {'tenant': 'north', 'total': 50})
        self.assertEqual(service.report(user, 'south'), {'tenant': 'south', 'total': 15})

    def test_cache_keeps_users_and_tenants_separate(self):
        service = ReportService(ReportStore({'north': [4], 'south': [9]}), ReportCache())
        first = {'id': 'first', 'tenants': {'north', 'south'}}
        second = {'id': 'second', 'tenants': {'south'}}
        self.assertEqual(service.report(first, 'north'), {'tenant': 'north', 'total': 4})
        self.assertEqual(service.report(first, 'south'), {'tenant': 'south', 'total': 9})
        self.assertEqual(service.report(second, 'south'), {'tenant': 'south', 'total': 9})

    def test_unauthorized_tenant_is_rejected_even_when_cached(self):
        service = ReportService(ReportStore({'north': [4]}), ReportCache())
        service.report({'id': 'analyst', 'tenants': {'north'}}, 'north')
        with self.assertRaises(PermissionError):
            service.report({'id': 'analyst', 'tenants': set()}, 'north')
