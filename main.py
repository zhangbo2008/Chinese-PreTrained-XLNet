
import tensorflow as tf


import numpy as np




def model_fn(features, labels, mode, params):
    #### Training or Evaluation
    is_training = False

    #### Get loss from inputs
    if FLAGS.is_regression:
      (total_loss, per_example_loss, logits
          ) = function_builder.get_regression_loss(FLAGS, features, is_training)
    else:
      (total_loss, per_example_loss, logits
          ) = function_builder.get_classification_loss(
          FLAGS, features, n_class, is_training)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    #### load pretrained models
    scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
      assert FLAGS.num_hosts == 1

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        eval_input_dict = {
            'labels': label_ids,
            'predictions': predictions,
            'weights': is_real_example
        }
        accuracy = tf.metrics.accuracy(**eval_input_dict)

        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            'eval_accuracy': accuracy,
            'eval_loss': loss}

      def regression_metric_fn(
          per_example_loss, label_ids, logits, is_real_example):
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        pearsonr = tf.contrib.metrics.streaming_pearson_correlation(
            logits, label_ids, weights=is_real_example)
        return {'eval_loss': loss, 'eval_pearsonr': pearsonr}

      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)

      #### Constucting evaluation TPUEstimatorSpec with new cache.
      label_ids = tf.reshape(features['label_ids'], [-1])

      if FLAGS.is_regression:
        metric_fn = regression_metric_fn
      else:
        metric_fn = metric_fn
      metric_args = [per_example_loss, label_ids, logits, is_real_example]

      if FLAGS.use_tpu:
        eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=(metric_fn, metric_args),
            scaffold_fn=scaffold_fn)
      else:
        eval_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=metric_fn(*metric_args))

      return eval_spec

    elif mode == tf.estimator.ModeKeys.PREDICT:
      label_ids = tf.reshape(features["label_ids"], [-1])

      predictions = {
          "logits": logits,
          "labels": label_ids,
          "is_real": features["is_real_example"]
      }

      if FLAGS.use_tpu:
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)
      return output_spec












estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config)
for pred_cnt, result in enumerate(estimator.predict(
        input_fn="pred_input_fn",
        yield_single_examples=True,
        checkpoint_path=FLAGS.predict_ckpt)):