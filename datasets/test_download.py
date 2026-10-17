"""Exercise download and failure handling without depending on the public mirror."""

import functools
import http.server
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import threading
import unittest


SCRIPT = Path(__file__).with_name("download.sh").resolve()
MATRICES = ("1138_bus", "494_bus", "662_bus", "685_bus")


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.requests = []
        requests = self.requests

        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                super().do_GET()

            def log_message(self, *_):
                pass

        self.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(Handler, directory=str(self.root))
        )
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()
        self.temporary.cleanup()

    def run_download(self):
        return subprocess.run(
            ["bash", str(SCRIPT), "suitsparsesmall"],
            cwd=self.root,
            env={
                **os.environ,
                "SUITESPARSE_MM_URL": f"http://127.0.0.1:{self.server.server_port}/MM",
            },
            text=True,
            capture_output=True,
            timeout=15,
        )

    def archives(self):
        folder = self.root / "MM" / "HB"
        folder.mkdir(parents=True)
        for name in MATRICES:
            data = b"%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 2\n"
            with tarfile.open(folder / f"{name}.tar.gz", "w:gz") as archive:
                member = tarfile.TarInfo(f"{name}/{name}.mtx")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    def test_success_extracts_matrices_and_reuses_completed_downloads(self):
        self.archives()
        # A directory left by an earlier failed run must not count as a dataset.
        (self.root / "output" / MATRICES[0]).mkdir(parents=True)
        result = self.run_download()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in MATRICES:
            target = self.root / "output" / name / f"{name}.mtx"
            self.assertTrue(target.read_text().startswith("%%MatrixMarket"))
        self.assertEqual(len(self.requests), len(MATRICES))
        self.assertFalse(list((self.root / "output").glob("*.tar.gz*")))
        result = self.run_download()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.requests), len(MATRICES))

    def test_http_failure_stops_before_extraction_and_cleans_partial_download(self):
        result = self.run_download()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.requests, ["/MM/HB/1138_bus.tar.gz"])
        self.assertFalse(list((self.root / "output").iterdir()))
        self.assertNotIn("tar:", result.stderr)

    def test_invalid_archive_is_reported_as_failure(self):
        self.archives()
        (self.root / "MM" / "HB" / "1138_bus.tar.gz").write_text("not an archive")
        result = self.run_download()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.requests, ["/MM/HB/1138_bus.tar.gz"])
        self.assertFalse(list((self.root / "output").rglob("*.mtx")))


if __name__ == "__main__":
    unittest.main()
