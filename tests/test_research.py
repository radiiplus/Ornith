from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from xing_specialist.backend import GenerationLimits
from xing_specialist.research import (ResearchError, ResearchLibrary, Source, browse,
                                      extract_findings, local_source, public_url)


class ResearchTests(unittest.TestCase):
    def test_source_quotes_scope_and_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library = ResearchLibrary(root / 'research.sqlite3')
            source_id = library.add_source(Source('global', 'https://example.com/a',
                                                  'A reproducible test verifies addition.'))
            library.add_finding(source_id, 'procedure', 'Verify addition',
                                'Run the reproducible test', 'reproducible test')
            with self.assertRaises(ResearchError):
                library.add_finding(source_id, 'claim', 'Unfounded', 'No evidence', 'missing quote')
            wrapped_id = library.add_source(Source('global', 'https://example.com/wrapped',
                                                   'Research verification.\n'
                                                   'A finding is accepted only when its quoted evidence\n'
                                                   'occurs exactly in the captured text.'))
            library.add_finding(wrapped_id, 'claim', 'Quote check', 'Evidence matches',
                                'A finding is accepted only when its quoted evidence '
                                'occurs exactly in the captured text.')
            saved_quote = library.db.execute('SELECT quote FROM research_findings '
                                             'WHERE source_id=?', (wrapped_id,)).fetchone()['quote']
            self.assertIn('evidence\noccurs', saved_quote)
            self.assertEqual(library.search_findings('research')[0]['source_id'], wrapped_id)
            self.assertEqual(len(library.search('addition')), 1)
            self.assertEqual(library.search_findings('addition')[0]['status'], 'source_checked')
            workspace = root / 'workspace'
            workspace.mkdir()
            (workspace / 'notes.txt').write_text('Local-only reduction procedure')
            local_id = library.add_source(local_source(workspace, 'notes.txt'))
            self.assertEqual(len(library.search('reduction', workspace)), 1)
            self.assertEqual(library.search('reduction'), [])
            self.assertNotEqual(local_id, source_id)
            library.add_source(Source('global', 'https://example.com/a', 'Different source text'))
            self.assertEqual(library.db.execute('SELECT COUNT(*) FROM research_findings '
                                                'WHERE source_id=?', (source_id,)).fetchone()[0], 0)
            self.assertEqual(library.search_findings('addition'), [])
            library.close()

    def test_browse_follows_relevant_same_host_links(self):
        address = [('family', None, None, None, ('93.184.215.14', 443))]
        pages = {
            'https://example.com/': ('Index', [('/addition', 'addition guide'),
                                                ('https://other.com/secret', 'addition external')]),
            'https://example.com/addition': ('Addition guide', [])}
        with patch('xing_specialist.research.socket.getaddrinfo', return_value=address):
            found = browse(('https://example.com/',), 'addition', 2,
                           fetch=lambda url: pages[url])
        self.assertEqual([row.locator for row in found],
                         ['https://example.com/', 'https://example.com/addition'])

    def test_rejects_private_hosts_and_unsafe_local_paths(self):
        with patch('xing_specialist.research.socket.getaddrinfo',
                   return_value=[('family', None, None, None, ('127.0.0.1', 443))]):
            with self.assertRaises(ResearchError):
                public_url('https://internal.example/')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ResearchError):
                local_source(Path(folder), '../private.txt')

    def test_model_extraction_accepts_only_exact_source_quotes(self):
        class FakeBackend:
            def generate(self, messages, limits):
                return '''{"findings":[
                    {"kind":"claim","title":"Parsed text","body":"Parser collects text",
                     "quote":"Parser collects text"},
                    {"kind":"procedure","title":"Invented","body":"Not present",
                     "quote":"fabricated source quote"}]}'''

        with tempfile.TemporaryDirectory() as folder:
            library = ResearchLibrary(Path(folder) / 'research.sqlite3')
            source_id = library.add_source(Source('global', 'https://example.com/',
                                                  'Parser collects text'))
            self.assertEqual(len(extract_findings(FakeBackend(), GenerationLimits(),
                                                  library, source_id)), 1)
            self.assertEqual(library.db.execute('SELECT COUNT(*) FROM research_findings').fetchone()[0], 1)
            audit = library.db.execute('SELECT accepted,rejected FROM research_extractions').fetchone()
            self.assertEqual((audit['accepted'], audit['rejected']), (1, 1))
            library.close()
