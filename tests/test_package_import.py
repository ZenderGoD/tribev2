import sys
import unittest


class PackageImportTests(unittest.TestCase):
    def test_importing_package_does_not_load_heavy_runtime_modules(self):
        sys.modules.pop("tribev2", None)
        sys.modules.pop("tribev2.demo_utils", None)

        import tribev2  # noqa: F401

        self.assertNotIn("tribev2.demo_utils", sys.modules)


if __name__ == "__main__":
    unittest.main()
