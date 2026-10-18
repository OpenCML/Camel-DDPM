"""Admission checks: an open metadata record must not admit restricted media."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("openaccess", Path(__file__).resolve().parents[1] / "tools/prepare_openaccess.py")
openaccess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(openaccess)


class AdmissionChecks(unittest.TestCase):
    def setUp(self):
        self.record = {"title": "A bird on a branch", "content": {
            "descriptiveNonRepeating": {"metadata_usage": {"access": "CC0"},
                "online_media": {"media": [{"type": "Images", "id": "media:test",
                    "usage": {"access": "CC0"},
                    "content": "https://ids.si.edu/ids/deliveryService?id=test"}]}}}}

    def test_explicit_metadata_and_image_license(self):
        self.assertEqual(len(list(openaccess.candidates(self.record))), 1)
        for access in (None, "", "CC-BY-NC", "restricted"):
            record = copy.deepcopy(self.record)
            record["content"]["descriptiveNonRepeating"]["online_media"]["media"][0]["usage"]["access"] = access
            self.assertEqual(list(openaccess.candidates(record)), [])

    def test_restricted_metadata(self):
        self.record["content"]["descriptiveNonRepeating"]["metadata_usage"]["access"] = "restricted"
        self.assertEqual(list(openaccess.candidates(self.record)), [])

    def test_missing_caption(self):
        self.record["title"] = ""
        self.assertEqual(list(openaccess.candidates(self.record)), [])

    def test_unrecognized_host_is_rejected_before_transport(self):
        with self.assertRaises(ValueError):
            openaccess.fetch("https://example.invalid/image.jpg")


if __name__ == "__main__":
    unittest.main()
