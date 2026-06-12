import tempfile
import unittest
from pathlib import Path

from src.kg.cmeie_hf import CMeIE_HF_FILES, build_hf_resolve_url, download_cmeie_files, download_file


class FakeResponse:
    def __init__(self, chunks):
        self.chunks = chunks
        self.status_checked = False

    def raise_for_status(self):
        self.status_checked = True

    def iter_content(self, chunk_size=1024 * 1024):
        yield from self.chunks


class FakeSession:
    def __init__(self):
        self.urls = []

    def get(self, url, stream=True, timeout=60):
        self.urls.append((url, stream, timeout))
        return FakeResponse([b'{"text":"a"}\n'])


class CMeIEHuggingFaceDownloadTests(unittest.TestCase):
    def test_build_hf_resolve_url_points_to_dataset_file(self):
        url = build_hf_resolve_url("CMeIE_train.jsonl")

        self.assertEqual(
            url,
            "https://huggingface.co/datasets/Aunderline/CMeIE/resolve/main/CMeIE_train.jsonl",
        )

    def test_download_file_writes_response_content_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "CMeIE_train.jsonl"
            response = FakeResponse([b"line1\n", b"line2\n"])

            downloaded = download_file(
                "https://example.com/CMeIE_train.jsonl",
                output_path,
                session=lambda url, stream, timeout: response,
            )

            self.assertEqual(downloaded, output_path)
            self.assertTrue(response.status_checked)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "line1\nline2\n")
            self.assertFalse(output_path.with_suffix(".jsonl.tmp").exists())

    def test_download_cmeie_files_skips_existing_files_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            existing_path = output_dir / "CMeIE_train.jsonl"
            existing_path.write_text("already here\n", encoding="utf-8")
            fake_session = FakeSession()

            paths = download_cmeie_files(
                output_dir,
                files=["CMeIE_train.jsonl", "CMeIE_dev.jsonl"],
                skip_existing=True,
                session_get=fake_session.get,
            )

            self.assertEqual([path.name for path in paths], ["CMeIE_train.jsonl", "CMeIE_dev.jsonl"])
            self.assertEqual(existing_path.read_text(encoding="utf-8"), "already here\n")
            self.assertEqual(len(fake_session.urls), 1)
            self.assertTrue((output_dir / "CMeIE_dev.jsonl").exists())

    def test_default_file_list_contains_train_dev_test_and_schema(self):
        self.assertEqual(
            CMeIE_HF_FILES,
            (
                "53_schemas.jsonl",
                "CMeIE_train.jsonl",
                "CMeIE_dev.jsonl",
                "CMeIE_test.jsonl",
            ),
        )


if __name__ == "__main__":
    unittest.main()
