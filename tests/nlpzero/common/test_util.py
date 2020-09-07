from unittest import TestCase

import numpy as np

from nlpzero.common import util


class PreprocessTestCase(TestCase):
    def test_return_numpy_array_as_corpus(self):
        text = "You say goodbye and I say hello."
        expected = np.array([0, 1, 2, 3, 4, 1, 5, 6])

        actual, _, _ = util.preprocess(text)

        np.testing.assert_array_equal(actual, expected)
