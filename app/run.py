"""Start the extraction API and Streamlit together: python -m app.run."""
import os
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from app.backend.extractor import ROOT


def main():
    processes = []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        backend = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'app.backend.main:app',
             '--host', '127.0.0.1', '--port', '8000'],
            cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', '4')},
        )
        processes.append(backend)
        readiness_opener = build_opener(ProxyHandler({}))
        deadline = time.monotonic() + 30
        while True:
            if backend.poll() is not None:
                raise RuntimeError('Extraction backend exited during startup. See the logs above.')
            try:
                with readiness_opener.open('http://127.0.0.1:8000/api/status', timeout=1) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError('Extraction backend did not become ready within 30 seconds.')
            time.sleep(0.2)
        frontend = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', 'app/frontend/streamlit_app.py',
             '--server.address', '127.0.0.1', '--server.port', '8501', '--server.headless', 'true'],
            cwd=ROOT, env={**os.environ, 'NERRE_API_URL': 'http://127.0.0.1:8000'},
        )
        processes.append(frontend)
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise RuntimeError('An application service exited. See the logs above.')
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
