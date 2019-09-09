# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Travis Yates

"""Tests for object_detection.utils.static_shape."""

import tensorflow as tf

from object_detection.utils import static_shape


class StaticShapeTest(tf.test.TestCase):

  def test_return_correct_batchSize(self):
    tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
    self.assertEqual(32, static_shape.get_batch_size(tensor_shape))

  def test_return_correct_height(self):
    tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
    self.assertEqual(299, static_shape.get_height(tensor_shape))

  def test_return_correct_width(self):
    tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
    self.assertEqual(384, static_shape.get_width(tensor_shape))

  def test_return_correct_depth(self):
    tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
    self.assertEqual(3, static_shape.get_depth(tensor_shape))

  def test_die_on_tensor_shape_with_rank_three(self):
    tensor_shape = tf.TensorShape(dims=[32, 299, 384])
    with self.assertRaises(ValueError):
      static_shape.get_batch_size(tensor_shape)
      static_shape.get_height(tensor_shape)
      static_shape.get_width(tensor_shape)
      static_shape.get_depth(tensor_shape)

if __name__ == '__main__':
  tf.test.main()
