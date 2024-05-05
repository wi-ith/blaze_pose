import numpy as np
import tensorflow as tf

class Loss():
    def __init__(self, threshold, n_threshold, num_samples, mode):
        self.mode = mode
        self.threshold = threshold
        self.n_threshold = n_threshold
        self.num_samples = num_samples
        self.record = {
            'n_threshold': n_threshold,
            'eq_correct_count': np.zeros(n_threshold),
            'not_eq_correct_count': np.zeros(n_threshold),
            'eq_count': 0,
            'not_eq_count': 0,
            'threshold': 0.1 * np.array(range(n_threshold)),
            'WKDR': np.zeros((n_threshold, 4))
        }
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        # self.reg_loss_fn = tf.keras.losses.Huber()
        self.reg_loss_fn = tf.keras.losses.MeanSquaredError()
        # self.loss_fn = tf.keras.losses.Huber()

    # def euclidean_distance_loss(self):
    #     # return tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))
    #
    #     return tf.keras.losses.MeanSquaredError()
    def calc_loss(self, batch_output, batch_heatmaps, batch_offsetmaps, batch_regression):
        if self.mode == 'train':
            # label = tf.cast(tf.round(label[:,:,:,0]/255), dtype=tf.int32)
            # print(batch_output.shape)
            heatmap_conf = batch_output['heatmap_logit']
            offset_y_pred = batch_output['offset_y']
            offset_x_pred = batch_output['offset_x']
            reg_pred = batch_output['reg_logit']
            conf_true = batch_heatmaps

            y_true = batch_offsetmaps[:, :, :, 0, :]
            x_true = batch_offsetmaps[:, :, :, 1, :]

            edl_loss = self.loss_fn(y_true = tf.cast(conf_true,dtype=tf.float32), y_pred = heatmap_conf)
            y_loss = self.loss_fn(y_true = tf.cast(y_true,dtype=tf.float32), y_pred =offset_y_pred)
            x_loss = self.loss_fn(y_true = tf.cast(x_true,dtype=tf.float32), y_pred =offset_x_pred)
            reg_loss = self.reg_loss_fn(y_true = batch_regression,y_pred = reg_pred)

            # edl_loss = tf.reduce_sum(edl_loss)
            # x_loss = tf.reduce_sum(x_loss)
            # y_loss = tf.reduce_sum(y_loss)

            loss_dict = {
                'edl_loss':edl_loss,
                'x_loss': x_loss,
                'y_loss':y_loss,
                'reg_loss':reg_loss
            }

            return loss_dict

