from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import dataloader_utils as dlu


class DistributedExecutor(object):
    def __init__(self, params, model_fn, eval_model_fn, loss_fn=None, eval_fn=None):
        self._params = params
        self._model_fn = model_fn
        self._eval_model_fn = eval_model_fn
        self._loss_fn = loss_fn
        self._eval_fn = eval_fn


    def loss_fn(self):
        return self._loss_fn()

    def eval_fn(self):
        return self._eval_fn()

    def model_fn(self, params):
        return self._model_fn(params)

    def eval_model_fn(self, params):
        return self._eval_model_fn(params)

    @tf.function(experimental_relax_shapes=True)
    def summary_fn(self, var_name, var, step):
        with self._summary_writer.as_default():
            tf.summary.scalar(var_name, var, step=step)

    @tf.function(experimental_relax_shapes=True)
    def summary_image_fn(self, summary_images, step, summary_name):
        with self._summary_writer.as_default():
            tf.summary.image(summary_name, summary_images, max_outputs=6, step=step)
            # tf.summary.image("input_image", summary_images[1], max_outputs=5, step=step)

    def _create_replicated_step(self,
                                model,
                                loss_fn,
                                optimizer
                                ):
        def _replicated_step(inputs, lr):
            images, labels = dlu.inputs(inputs,self._params)

            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                prediction_loss = loss_fn(labels, outputs)
                loss = tf.reduce_mean(prediction_loss)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        return _replicated_step


    def _create_train_step(self,
                           model,
                           loss_fn,
                           optimizer,
                           metric=None):

        replicated_step = self._create_replicated_step(model, loss_fn,
                                                       optimizer, metric)
        @tf.function
        def train_step(iterator, epoch, lr):
            for step, parsed_record in enumerate(iterator):
                losses = replicated_step(parsed_record, lr)
                cur_step = self._params['train']['train_samples']*epoch + step
            return losses

        return train_step


    def train(self,
              train_input_fn,
              val_input_fn
              ):
        total_epochs = self._params['train']['total_epochs']
        model = self.model_fn(self._params)
        if self._params['train']['load_weight_path']is not None:
            model.load_weights(filepath=self._params['train']['load_weight_path'])
            print('load trained weights')

        eval_model = self.eval_model_fn(self._params)
        optimizer = model.optimizer
        train_step = self._create_train_step(
            model = model,
            loss_fn=self.loss_fn(),
            optimizer = optimizer)

        _lr = self._params['train']['learning_rate']
        for epoch in range(total_epochs):
            updated_lr = _lr['init_learning_rate'] * ((_lr['learning_rate_decay_rate']) ** (epoch // _lr['learning_rate_decay_epochs']))
            train_step(train_input_fn, epoch, updated_lr)
            ckpt_path = self._params['train']['model_dir']+"{}-epoch".format(epoch)
            model.save_weights(filepath=ckpt_path)
            eval_model.load_weights(filepath=ckpt_path)
            self.eval(val_input_fn,eval_model,epoch)

    def eval(self,
             val_input_fn,
             model,
             epoch
             ):
        test_step = self._create_test_step(eval_fn = self.eval_fn(), model = model)
        test_step(val_input_fn, epoch)
