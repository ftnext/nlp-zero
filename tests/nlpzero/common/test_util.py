from unittest import TestCase

import numpy as np

from nlpzero.common import util


class PreprocessTestCase(TestCase):
    def test_return_numpy_array_as_corpus(self):
        text = "You say goodbye and I say hello."
        expected = np.array([0, 1, 2, 3, 4, 1, 5, 6])

        actual, _, _ = util.preprocess(text)

        np.testing.assert_array_equal(actual, expected)


class CreateCoMatrixTestCase(TestCase):
    def test_return_numpy_array_as_cooccurrence_matrix(self):
        text = "You say goodbye and I say hello."
        corpus, word_to_id, _ = util.preprocess(text)
        expected = np.array(
            [
                [0, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 1, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=np.int32,
        )

        actual = util.create_co_matrix(corpus, len(word_to_id))

        np.testing.assert_array_equal(actual, expected)
