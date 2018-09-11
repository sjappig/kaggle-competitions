import unittest

import numpy

import work.common as common

# pylint: disable=no-self-use
class TestCommon(unittest.TestCase):
    def test_encoding_from_mask(self):
        original_mask = numpy.array([[1, 0, 0], [0, 1, 0]])
        encoded = common.mask_to_encoded_pixels(original_mask)
        mask = common.encoded_pixels_to_mask(encoded, original_mask.shape)

        numpy.testing.assert_array_equal(mask, original_mask)

    def test_encoding_from_decimalmask(self):
        original_mask = numpy.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0]])
        encoded = common.mask_to_encoded_pixels(original_mask, threshold=0.3)
        mask = common.encoded_pixels_to_mask(encoded, original_mask.shape)

        numpy.testing.assert_array_equal(
            mask,
            numpy.array([[0, 0, 1], [1, 0, 0]]),
        )

    def test_encoding_from_encoded(self):
        original_encoded = '1 2 4 7 100 1'
        mask = common.encoded_pixels_to_mask(original_encoded, (50, 50))
        encoded = common.mask_to_encoded_pixels(mask)

        numpy.testing.assert_array_equal(original_encoded, encoded)
