import io
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile

from app.backend import main as server
from training import eval as evaluation


class RepairTests(unittest.TestCase):
    def test_loaded_extractor_does_not_wait_for_generation(self):
        model = object()
        with patch.object(server, '_extractor', model), ThreadPoolExecutor() as pool:
            with server._generation_lock:
                future = pool.submit(server.get_extractor)
                self.assertIs(future.result(timeout=2), model)

    def test_concurrent_initialization_loads_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'config.json').write_text('{}')
            with patch.object(server, 'MODEL_PATH', path), patch.object(server, '_extractor', None), patch.object(server, 'Extractor') as factory:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    models = list(pool.map(lambda _: server.get_extractor(), range(8)))
                factory.assert_called_once_with(path)
                self.assertTrue(all(model is factory.return_value for model in models))

    def test_generation_remains_serialized(self):
        class Model:
            active = peak = 0
            guard = threading.Lock()

            def extract(self, sentence):
                with self.guard:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                try:
                    time.sleep(0.02)
                    return {'basemats': {}, 'dopants': {}, 'dopants2basemats': {}}
                finally:
                    with self.guard:
                        self.active -= 1
        model = Model()
        def request(_):
            return server.extract(UploadFile(filename='doc.txt', file=io.BytesIO(b'Sentence.')))
        with patch.object(server, 'get_extractor', return_value=model), ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(request, range(4)))
        self.assertEqual(model.peak, 1)
        self.assertTrue(all(result['status'] == 'complete' for result in results))

    def test_duplicate_names_collapse_without_normalizing_strings(self):
        graph = {'basemats': {'b0': 'ZnO', 'b1': 'ZnO'}, 'dopants': {'d0': 'Al'}, 'dopants2basemats': {'d0': ['b0', 'b1']}}
        self.assertEqual(evaluation.graph_sets(graph), {'basemats': {'ZnO'}, 'dopants': {'Al'}, 'links': {('Al', 'ZnO')}})
        graph['basemats']['b1'] = ' ZnO'
        self.assertEqual(evaluation.graph_sets(graph)['basemats'], {'ZnO', ' ZnO'})

    def test_dangling_gold_reference_fails_before_model_loading(self):
        for links in ({'d0': ['missing']}, {'missing': ['b0']}):
            with self.subTest(links=links), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'gold.json'
                path.write_text(json.dumps([{'doping_sentences': [{'sentence_text': 'Al-doped ZnO.', 'relevant': True, 'basemats': {'b0': 'ZnO'}, 'dopants': {'d0': 'Al'}, 'dopants2basemats': links}]}]))
                with patch('sys.argv', ['eval', '--data', str(path)]), patch.object(evaluation, 'Extractor') as factory:
                    with self.assertRaisesRegex(ValueError, 'Invalid gold graph for sentence 1:.*invalid entity links'):
                        evaluation.main()
                    factory.assert_not_called()
