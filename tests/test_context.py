from pathlib import Path
import json
import tempfile
import unittest

from xing_specialist.agent import EngineeringAgent, SYSTEM_PROMPT, build_model_prompt, parse_proposal
from xing_specialist.backend import GenerationLimits
from xing_specialist.context import ContextRouter
from xing_specialist.memory import MemoryStore


class SearchBackend:
    def generate(self, messages, limits):
        return json.dumps({'note': '[SOURCE notes.txt chunk 0] points to the parser',
                           'next_queries': ['thread deadlock']})


class UncitedBackend:
    def generate(self, messages, limits):
        return json.dumps({'note': 'The code is definitely wrong', 'next_queries': []})


class QuestionBackend:
    def generate(self, messages, limits):
        return json.dumps({'note': '[SOURCE notes.txt chunk 0] shows a parser failure',
                           'next_queries': [],
                           'open_questions': ['Does this also fail with an empty input?']})


class ContextRouterTests(unittest.TestCase):
    def test_reasoning_block_before_json(self):
        self.assertEqual(parse_proposal('<think>Search first.</think>\n{"next_queries": []}'),
                         {'next_queries': []})

    def test_complete_json_in_unclosed_fence(self):
        self.assertEqual(parse_proposal('```json\n{"edits": []}\n'), {'edits': []})
        with self.assertRaises(ValueError):
            parse_proposal('```json\n{"edits": []}\nextra text')

    def test_routes_relevant_chunks_with_a_prompt_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'notes.txt').write_text('alpha baseline ' * 300 + 'nested parse regression ' * 30)
            router = ContextRouter(root / 'context.sqlite3')
            try:
                self.assertGreater(router.index_file(repo, 'notes.txt'), 1)
                routed = router.route(repo, 'nested parse regression', 1200)
                self.assertIn('nested parse regression', routed)
                self.assertIn('[SOURCE notes.txt chunk ', routed)
                self.assertLessEqual(len(routed), 1200)
                prompt, selected = build_model_prompt(
                    'Fix nested parse regression', 'FILE calc.py\npass', 'baseline fails', [],
                    ('calc.py',), GenerationLimits(2048, 512, 5), router, repo)
                self.assertIn(selected, prompt)
                self.assertTrue(selected)
                self.assertLessEqual(len(prompt) + len(SYSTEM_PROMPT), (2048 - 512) * 3)
                large_prompt, _ = build_model_prompt(
                    'Fix nested parse regression', 'FILE calc.py\n' + 'x' * 16000,
                    'baseline fails' * 500, [], ('calc.py',), GenerationLimits(2048, 512, 5),
                    router, repo)
                self.assertLessEqual(len(large_prompt) + len(SYSTEM_PROMPT), (2048 - 512) * 3)
            finally:
                router.close()

    def test_repair_prompt_keeps_editable_code_and_rationale_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'test_calc.py').write_text('assert add(2, 3) == 5\n')
            router = ContextRouter(root / 'context.sqlite3')
            try:
                router.index_file(repo, 'test_calc.py')
                prompt, routed = build_model_prompt(
                    'Fix add so the test passes', 'FILE calc.py\ndef add(a, b):\n    return a - b\n',
                    'Baseline: add(2, 3) returned -1, expected 5.', [], ('calc.py',),
                    GenerationLimits(2048, 384, 90), router, repo,
                    context_sources=('test_calc.py',))
                self.assertIn('return a - b', prompt)
                self.assertIn('returned -1, expected 5', prompt)
                self.assertIn('Return only JSON with hypothesis, expected, and edits', prompt)
                self.assertIn('[SOURCE test_calc.py chunk 0]', routed)
                self.assertLessEqual(len(prompt) + len(SYSTEM_PROMPT), (2048 - 384) * 3)
            finally:
                router.close()

    def test_reindex_replaces_old_chunks_and_scopes_repositories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = root / 'first'
            second = root / 'second'
            first.mkdir()
            second.mkdir()
            (first / 'notes.txt').write_text('unique_alpha ' * 100)
            (second / 'notes.txt').write_text('unique_beta ' * 100)
            router = ContextRouter(root / 'context.sqlite3')
            try:
                router.index_file(first, 'notes.txt')
                router.index_file(second, 'notes.txt')
                self.assertEqual(router.route(first, 'unique_beta', 1200), '')
                self.assertEqual(router.route(second, 'unique_beta', 1200, sources=('other.txt',)), '')
                (first / 'notes.txt').write_text('unique_gamma ' * 100)
                router.index_file(first, 'notes.txt')
                self.assertEqual(router.route(first, 'unique_alpha', 1200), '')
                self.assertIn('unique_gamma', router.route(first, 'unique_gamma', 1200))
            finally:
                router.close()

    def test_unchanged_index_is_reused_and_same_size_change_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / 'repo'
            repo.mkdir()
            source = repo / 'notes.txt'
            source.write_text('alpha ' * 200)
            router = ContextRouter(root / 'context.sqlite3')
            try:
                count = router.index_file(repo, 'notes.txt')
                rowids = [row[0] for row in router.db.execute('SELECT rowid FROM context_fts')]
                self.assertEqual(router.index_file(repo, 'notes.txt'), count)
                self.assertTrue(router.last_index_reused)
                self.assertEqual(rowids, [row[0] for row in router.db.execute('SELECT rowid FROM context_fts')])
                source.write_text('bravo ' * 200)
                router.index_file(repo, 'notes.txt')
                self.assertFalse(router.last_index_reused)
                self.assertEqual(router.route(repo, 'alpha', 1200), '')
                self.assertIn('bravo', router.route(repo, 'bravo', 1200))
            finally:
                router.close()

    def test_route_emits_only_selected_chunk_events(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'notes.txt').write_text('parser failure ' * 80)
            events = []
            router = ContextRouter(root / 'context.sqlite3',
                                   on_chunk=lambda *parts: events.append(parts))
            try:
                router.index_file(repo, 'notes.txt')
                routed = router.route(repo, 'parser failure', 900, max_chunks=1)
                self.assertIn('[SOURCE notes.txt chunk', routed)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0][0], 'notes.txt')
            finally:
                router.close()

    def test_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / 'repo'
            repo.mkdir()
            (root / 'outside.txt').write_text('secret')
            router = ContextRouter(root / 'context.sqlite3')
            try:
                with self.assertRaises(ValueError):
                    router.index_file(repo, '../outside.txt')
            finally:
                router.close()

    def test_agent_queues_a_followup_search(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'notes.txt').write_text('parser failure ' * 80 + 'thread deadlock ' * 80)
            store = MemoryStore(root / 'memory.sqlite3')
            events = []
            router = ContextRouter(root / 'context.sqlite3',
                                   on_chunk=lambda *parts: events.append(('context_chunk', *parts)))
            try:
                router.index_file(repo, 'notes.txt')
                task_id = store.task('Fix parser failure')
                agent = EngineeringAgent(SearchBackend(), store, None,
                                         GenerationLimits(2048, 512, 5), context_router=router,
                                         on_event=lambda *parts: events.append(parts))
                query, notes = agent.investigate_context(task_id, 'Fix parser failure', repo, ('notes.txt',))
                self.assertEqual(query, 'thread deadlock')
                self.assertIn('[SOURCE notes.txt chunk 0]', notes)
                kinds = [row[0] for row in events]
                note_index = kinds.index('research_note')
                self.assertIn('context_chunk', kinds[:note_index])
                self.assertIn('context_chunk', kinds[note_index + 1:])
                self.assertEqual(store.db.execute(
                    "SELECT COUNT(*) FROM actions WHERE kind='context_search'").fetchone()[0], 2)
            finally:
                router.close()
                store.close()

    def test_uncited_working_note_is_discarded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'notes.txt').write_text('parser failure ' * 80)
            store = MemoryStore(root / 'memory.sqlite3')
            router = ContextRouter(root / 'context.sqlite3')
            try:
                router.index_file(repo, 'notes.txt')
                agent = EngineeringAgent(UncitedBackend(), store, None,
                                         GenerationLimits(2048, 512, 5), context_router=router)
                _, notes = agent.investigate_context(store.task('Fix parser'), 'Fix parser', repo, ('notes.txt',))
                self.assertEqual(notes, '')
            finally:
                router.close()
                store.close()

    def test_research_question_requires_recurrence_before_recall(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'notes.txt').write_text('parser failure ' * 80)
            store = MemoryStore(root / 'memory.sqlite3')
            router = ContextRouter(root / 'context.sqlite3')
            try:
                router.index_file(repo, 'notes.txt')
                agent = EngineeringAgent(QuestionBackend(), store, None,
                                         GenerationLimits(2048, 512, 5), context_router=router)
                for expected_count in (0, 1):
                    agent.investigate_context(store.task('Fix parser failure'),
                                              'Fix parser failure', repo, ('notes.txt',))
                    self.assertEqual(len(agent.adaptive.recall(str(repo), 'Fix parser failure')),
                                     expected_count)
                self.assertIn('empty input', agent.adaptive.recall(str(repo), 'Fix parser failure')[0]['body'])
            finally:
                router.close()
                store.close()


if __name__ == '__main__':
    unittest.main()
