import unittest

from packaging.tags import Tag

from scripts.cl5_dependency_budget import (
    MISSING, choose_compatible_wheel, report_budget,
)


def wheel(filename, size, *, yanked=False, requires_python=None):
    return {"filename": filename, "size": size, "packagetype": "bdist_wheel",
            "yanked": yanked, "requires_python": requires_python}


class DependencyBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tags = {Tag("py3", "none", "any")}

    def test_uses_latest_allowed_release_and_compatible_wheel(self):
        versions = {
            "1.0.0": [wheel("example-1.0.0-py3-none-any.whl", 100)],
            "2.0.0": [wheel("example-2.0.0-py3-none-any.whl", 200)],
            "3.0.0": [wheel("example-3.0.0-py3-none-any.whl", 100)],
        }
        chosen = choose_compatible_wheel(versions, ">=1,<3", self.tags)
        self.assertEqual(chosen["version"], "2.0.0")
        self.assertEqual(chosen["wheel_bytes"], 200)

    def test_ignores_incompatible_and_yanked_wheels(self):
        versions = {"1.0.0": [
            wheel("example-1.0.0-cp313-cp313-win_amd64.whl", 100),
            wheel("example-1.0.0-py3-none-any.whl", 200, yanked=True),
        ]}
        self.assertIsNone(choose_compatible_wheel(versions, "", self.tags))

    def test_requires_python_excludes_latest_release_for_proxy_python311(self):
        versions = {
            "1.0.0": [wheel("example-1.0.0-py3-none-any.whl", 100,
                            requires_python=">=3.9")],
            "2.0.0": [wheel("example-2.0.0-py3-none-any.whl", 200,
                            requires_python=">=3.12")],
        }
        self.assertEqual(choose_compatible_wheel(versions, "", self.tags)["version"], "1.0.0")
        self.assertIsNone(choose_compatible_wheel(
            {"2.0.0": versions["2.0.0"]}, "", self.tags))

    def test_unknown_package_never_approves_complete_fit(self):
        releases = {name: {"99.0.0": [wheel(f"{name.replace('-', '_')}-99.0.0-py3-none-any.whl", 123)]}
                    for name in MISSING}
        report = report_budget(releases, {"gguf": "HTTP404"}, self.tags)
        self.assertIn("gguf", report["unknown_wheel_packages"])
        self.assertIsNone(report["direct_wheels_fit_estimated_margin"])
        self.assertFalse(report["model_downloaded"])
        self.assertFalse(report["gpu_used"])

    def test_known_sizes_are_only_download_estimate(self):
        releases = {name: {"99.0.0": [wheel(f"{name.replace('-', '_')}-99.0.0-py3-none-any.whl", 123)]}
                    for name in MISSING}
        report = report_budget(releases, {}, self.tags)
        self.assertEqual(report["direct_wheel_bytes_known"], 123 * 6)
        # Pinned packages and the pydantic-settings ~=2.0 constraint reject 99.0.0.
        self.assertTrue(report["unknown_wheel_packages"])
        self.assertFalse(report["dependency_resolution_performed"])
