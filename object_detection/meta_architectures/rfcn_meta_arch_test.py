# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Travis Yates

"""Tests for object_detection.meta_architectures.rfcn_meta_arch."""

import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch_test_lib
from object_detection.meta_architectures import rfcn_meta_arch


class RFCNMetaArchTest(
    faster_rcnn_meta_arch_test_lib.FasterRCNNMetaArchTestBase):

  def _get_second_stage_box_predictor_text_proto(self):
    box_predictor_text_proto = """
      rfcn_box_predictor {
        conv_hyperparams {
          op: CONV
          activation: NONE
          regularizer {
            l2_regularizer {
              weight: 0.0005
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
      }
    """
    return box_predictor_text_proto

  def _get_model(self, box_predictor, **common_kwargs):
    return rfcn_meta_arch.RFCNMetaArch(
        second_stage_rfcn_box_predictor=box_predictor, **common_kwargs)


if __name__ == '__main__':
  tf.test.main()
