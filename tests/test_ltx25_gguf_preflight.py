import unittest

from scripts.ltx25_gguf_preflight import COMPONENTS, PACKS, REPOS, evaluate


def metadata(size=1_000_000_000):
    files = {key: {} for key in REPOS}
    for key, (repo_key, basename) in COMPONENTS.items():
        files[repo_key][f"folder/{basename}"] = size
    return files


class GGUFPreflightTests(unittest.TestCase):
    def setUp(self):
        self.revisions = {key: "abc" for key in REPOS}
        self.gated = {key: key == "encoder" for key in REPOS}

    def test_small_packs_are_disk_only_not_runtime_approved(self):
        result = evaluate(metadata(), self.revisions, self.gated)
        self.assertFalse(result["model_downloaded"])
        self.assertFalse(result["gpu_used"])
        self.assertTrue(result["repositories"]["encoder"]["gated"])
        for key in PACKS:
            self.assertTrue(result["packs"][key]["weights_fit_disk_only"])
            self.assertEqual(result["packs"][key]["runtime_feasibility"], "not tested")

    def test_oversized_pack_does_not_fit(self):
        result = evaluate(metadata(9_000_000_000), self.revisions, self.gated)
        for pack in result["packs"].values():
            self.assertFalse(pack["weights_fit_disk_only"])
            self.assertLess(pack["bytes_remaining_after_reserve"], 0)

    def test_missing_or_duplicate_component_fails_closed(self):
        files = metadata()
        del files["encoder"]["folder/gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf"]
        with self.assertRaises(ValueError):
            evaluate(files, self.revisions, self.gated)
        files = metadata()
        files["encoder"]["elsewhere/gemma4-12b-with-proj-ltx-2.5-Q2_K.gguf"] = 1
        with self.assertRaises(ValueError):
            evaluate(files, self.revisions, self.gated)

    def test_invalid_size_rejected(self):
        files = metadata()
        files["vae"]["folder/ltx-2.5-audio-vae-bf16.safetensors"] = None
        with self.assertRaises(ValueError):
            evaluate(files, self.revisions, self.gated)
