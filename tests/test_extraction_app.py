import asyncio
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from app.backend import main as server
from training.eval import graph_sets
from app.backend.extractor import parse_prediction, prompt_for, split_sentences
from training.finetune import read_examples, save_checkpoint


class FakeExtractor:
    def extract(self, sentence):
        if 'INVALID' in sentence:
            raise ValueError('Model returned invalid structured JSON')
        return {'basemats': {'b0': 'ZnO'}, 'dopants': {'d0': 'Al'}, 'dopants2basemats': {'d0': ['b0']}}


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def test_training_data_matches_inference_prompt_and_schema(self):
        examples = read_examples(Path(__file__).resolve().parents[1] / 'data/labeled/training_json.jsonl')
        self.assertEqual(len(examples), 413)
        self.assertEqual(examples[0][0], prompt_for('We report the properties of upconversion photoluminescence (UCPL) on SrLaMgTaO6 crystal doped with trivalent erbium at concentrations of 1–20mol% that were synthesized by a solid state reaction method from powder based precursors.'))
        self.assertEqual(parse_prediction(examples[0][1])['dopants']['d0'], 'trivalent erbium')

    def test_schema_rejects_invalid_json_and_links(self):
        for value in ['not JSON', '{"basemats":{},"dopants":{},"dopants2basemats":{"d0":["b0"]}}', '{"basemats":[],"dopants":{},"dopants2basemats":{}}']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_prediction(value)

    def test_sentence_splitting(self):
        self.assertEqual(split_sentences('Al-doped ZnO. Next sentence!\nA heading'), ['Al-doped ZnO.', 'Next sentence!', 'A heading'])
        self.assertEqual(split_sentences('Al-doped\nZnO is studied. Next result.\n\nSection heading'), ['Al-doped ZnO is studied.', 'Next result.', 'Section heading'])
        self.assertEqual(split_sentences('Doped films, i.e. ZnO:Al, were grown. See Fig. 2.\n\nNext section.'), ['Doped films, i.e. ZnO:Al, were grown.', 'See Fig. 2.', 'Next section.'])
        self.assertEqual(split_sentences('ZnO. Next sentence.'), ['ZnO.', 'Next sentence.'])
        self.assertEqual(split_sentences('Doped with Ca. Next. Element No. Next. See No. 2.'), ['Doped with Ca.', 'Next.', 'Element No.', 'Next.', 'See No. 2.'])

    @patch('app.backend.main.get_extractor', return_value=FakeExtractor())
    def test_upload_returns_sentence_level_json(self, _):
        response = self.client.post('/api/extract', files={'file': ('research.txt', b'Al-doped ZnO. Another sentence.', 'text/plain')})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['filename'], 'research.txt')
        self.assertEqual(body['status'], 'complete')
        self.assertEqual(len(body['sentences']), 2)
        self.assertEqual(body['sentences'][0]['dopants2basemats'], {'d0': ['b0']})

    @patch('app.backend.main.get_extractor', return_value=FakeExtractor())
    def test_invalid_model_output_is_reported_as_partial(self, _):
        response = self.client.post('/api/extract', files={'file': ('x.txt', b'Good. INVALID.', 'text/plain')})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['status'], 'partial')
        self.assertEqual(body['sentences'][0]['status'], 'complete')
        self.assertEqual(body['sentences'][1]['status'], 'error')
        self.assertIn('invalid structured JSON', body['sentences'][1]['error'])

    @patch('app.backend.main.get_extractor', return_value=FakeExtractor())
    def test_pdf_upload_extracts_text(self, _):
        writer = PdfWriter()
        page = writer.add_blank_page(width=200, height=200)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 20 100 Td (Al-doped ZnO.) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        pdf = io.BytesIO()
        writer.write(pdf)
        response = self.client.post('/api/extract', files={'file': ('paper.pdf', pdf.getvalue(), 'application/pdf')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['sentences'][0]['sentence'], 'Al-doped ZnO.')

    def test_pdf_page_limit_is_checked_before_extraction(self):
        writer = PdfWriter()
        for _ in range(server.MAX_PDF_PAGES + 1):
            writer.add_blank_page(width=100, height=100)
        pdf = io.BytesIO()
        writer.write(pdf)
        response = self.client.post('/api/extract', files={'file': ('paper.pdf', pdf.getvalue(), 'application/pdf')})
        self.assertEqual(response.status_code, 400)
        self.assertIn('page limit', response.json()['detail'])

    def test_evaluation_ignores_arbitrary_entity_ids(self):
        first = {'basemats': {'b0': 'ZnO'}, 'dopants': {'d0': 'Al'}, 'dopants2basemats': {'d0': ['b0']}}
        second = {'basemats': {'b1': 'ZnO'}, 'dopants': {'d1': 'Al'}, 'dopants2basemats': {'d1': ['b1']}}
        self.assertEqual(graph_sets(first), graph_sets(second))

    def test_bad_documents_are_rejected(self):
        for name, body, code in [('x.doc', b'hello', 400), ('x.txt', b'\xff', 400), ('x.txt', b'  ', 400), ('x.pdf', b'not pdf', 400), ('x.txt', b'a' * (server.MAX_BYTES + 1), 413), ('x.txt', b'word. ' * 101, 400)]:
            with self.subTest(name=name, size=len(body)):
                response = self.client.post('/api/extract', files={'file': (name, body)})
                self.assertEqual(response.status_code, code)

    def test_checkpoint_is_published_only_after_all_writes_succeed(self):
        class Model:
            def save_pretrained(self, path):
                (path / 'config.json').write_text('{}')
        class Tokenizer:
            def __init__(self, fail):
                self.fail = fail
            def save_pretrained(self, path):
                if self.fail:
                    raise OSError('write failed')
                (path / 'tokenizer.json').write_text('{}')
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'model'
            with self.assertRaises(OSError):
                save_checkpoint(Model(), Tokenizer(True), target, {})
            self.assertFalse(target.exists())
            save_checkpoint(Model(), Tokenizer(False), target, {'epochs': 1})
            self.assertTrue((target / 'config.json').is_file())
            self.assertTrue((target / 'tokenizer.json').is_file())
            with self.assertRaises(FileExistsError):
                save_checkpoint(Model(), Tokenizer(False), target, {})

    def test_streamed_upload_limit_before_multipart_parser(self):
        async def exercise():
            invoked = False
            async def downstream(scope, receive, send):
                nonlocal invoked
                invoked = True
            middleware = server.UploadLimitMiddleware(downstream)
            messages = iter([{'type': 'http.request', 'body': b'x' * (server.MAX_BYTES + 65537), 'more_body': False}])
            sent = []
            async def receive():
                return next(messages)
            async def send(message):
                sent.append(message)
            scope = {'type': 'http', 'method': 'POST', 'path': '/api/extract', 'headers': []}
            await middleware(scope, receive, send)
            self.assertFalse(invoked)
            self.assertEqual(sent[0]['status'], 413)
        asyncio.run(exercise())

    @patch('app.backend.main.MODEL_PATH', Path('/nonexistent/model'))
    def test_missing_model_is_reported(self):
        with patch('app.backend.main._extractor', None):
            response = self.client.post('/api/extract', files={'file': ('x.txt', b'One sentence.')})
            self.assertEqual(response.status_code, 503)
            self.assertIn('fine-tuned model', response.json()['detail'])


if __name__ == '__main__':
    unittest.main()
