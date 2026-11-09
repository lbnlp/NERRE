import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app/frontend/streamlit_app.py'


def response(payload, ok=True, status=200):
    result = Mock()
    result.ok = ok
    result.status_code = status
    result.json.return_value = payload
    return result


class StreamlitTests(unittest.TestCase):
    @patch('requests.post')
    @patch('requests.get')
    def test_sample_extracts_and_marks_partial(self, get, post):
        get.return_value = response({'model_ready': True})
        post.return_value = response({'filename': 'sample.txt', 'schema': 'doping', 'status': 'partial', 'sentences': [{'sentence': 'test', 'status': 'error', 'error': 'invalid JSON'}]})
        app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.button[1].disabled)
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertIn('1 sentence', app.warning[0].value)
        self.assertEqual(post.call_args.kwargs['files']['file'][0], 'sample.txt')
        self.assertEqual(len(app.json), 1)

    @patch('requests.post')
    @patch('requests.get')
    def test_backend_error_is_visible(self, get, post):
        get.return_value = response({'model_ready': False})
        post.return_value = response({'detail': 'No fine-tuned model found.'}, ok=False, status=503)
        app = AppTest.from_file(str(APP)).run()
        app.button[0].click().run()
        app.button[1].click().run()
        self.assertIn('No fine-tuned model', app.error[0].value)


if __name__ == '__main__':
    unittest.main()
