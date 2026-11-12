"""The local readiness request must ignore environment proxy configuration."""
import io
import os
import unittest
from email.message import Message
from unittest.mock import Mock, patch
from urllib.response import addinfourl
from urllib.request import Request, build_opener

from app import run


class LauncherTests(unittest.TestCase):
    def test_readiness_ignores_proxy_and_starts_frontend(self):
        backend, frontend = Mock(), Mock()
        backend.poll.return_value = frontend.poll.return_value = None
        response = addinfourl(io.BytesIO(b'{}'), Message(),
                              'http://127.0.0.1:8000/api/status', 200)
        response.msg = 'OK'
        proxy_env = {'http_proxy': 'http://proxy.invalid:3128',
                     'HTTP_PROXY': 'http://proxy.invalid:3128',
                     'no_proxy': '', 'NO_PROXY': ''}
        with patch.dict(os.environ, proxy_env, clear=True):
            # Ensure this fixture actually activates default proxy handling.
            request = Request('http://127.0.0.1:8000/api/status')
            with patch('urllib.request.HTTPHandler.http_open', return_value=response):
                build_opener().open(request, timeout=1)
            self.assertEqual(request.host, 'proxy.invalid:3128')
            with patch.object(run.subprocess, 'Popen', side_effect=[backend, frontend]) as popen, \
                 patch.object(run.signal, 'signal'), \
                 patch.object(run.time, 'sleep', side_effect=KeyboardInterrupt), \
                 patch('urllib.request.ProxyHandler.proxy_open', side_effect=AssertionError('Proxy used')), \
                 patch('urllib.request.HTTPHandler.http_open') as http_open:
                http_open.return_value = addinfourl(io.BytesIO(b'{}'), Message(), request.full_url, 200)
                http_open.return_value.msg = 'OK'
                run.main()
            self.assertEqual(popen.call_count, 2)
            self.assertIn('streamlit', popen.call_args_list[1].args[0])
            actual = http_open.call_args.args[0]
            self.assertEqual(actual.full_url, 'http://127.0.0.1:8000/api/status')
            self.assertEqual(actual.timeout, 1)
            backend.terminate.assert_called_once()
            frontend.terminate.assert_called_once()
