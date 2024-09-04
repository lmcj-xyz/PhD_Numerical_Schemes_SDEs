import unittest

from dsdes import fbm
import numpy as np


class TestFBM(unittest.TestCase):
    def test_type(self):
        self.assertIsInstance(fbm(0.75, 100, 5), np.ndarray)

    def test_size(self):
        self.assertEqual(fbm(0.75, 1000, 5).shape, (1000,))


if __name__ == "__main__":
    unittest.main()
