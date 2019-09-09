# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Travis Yates

"""Tests for region_similarity_calculator_builder."""

import tensorflow as tf

from google.protobuf import text_format
from object_detection.builders import region_similarity_calculator_builder
from object_detection.core import region_similarity_calculator
from object_detection.protos import region_similarity_calculator_pb2 as sim_calc_pb2


class RegionSimilarityCalculatorBuilderTest(tf.test.TestCase):

  def testBuildIoaSimilarityCalculator(self):
    similarity_calc_text_proto = """
      ioa_similarity {
      }
    """
    similarity_calc_proto = sim_calc_pb2.RegionSimilarityCalculator()
    text_format.Merge(similarity_calc_text_proto, similarity_calc_proto)
    similarity_calc = region_similarity_calculator_builder.build(
        similarity_calc_proto)
    self.assertTrue(isinstance(similarity_calc,
                               region_similarity_calculator.IoaSimilarity))

  def testBuildIouSimilarityCalculator(self):
    similarity_calc_text_proto = """
      iou_similarity {
      }
    """
    similarity_calc_proto = sim_calc_pb2.RegionSimilarityCalculator()
    text_format.Merge(similarity_calc_text_proto, similarity_calc_proto)
    similarity_calc = region_similarity_calculator_builder.build(
        similarity_calc_proto)
    self.assertTrue(isinstance(similarity_calc,
                               region_similarity_calculator.IouSimilarity))

  def testBuildNegSqDistSimilarityCalculator(self):
    similarity_calc_text_proto = """
      neg_sq_dist_similarity {
      }
    """
    similarity_calc_proto = sim_calc_pb2.RegionSimilarityCalculator()
    text_format.Merge(similarity_calc_text_proto, similarity_calc_proto)
    similarity_calc = region_similarity_calculator_builder.build(
        similarity_calc_proto)
    self.assertTrue(isinstance(similarity_calc,
                               region_similarity_calculator.
                               NegSqDistSimilarity))


if __name__ == '__main__':
  tf.test.main()
