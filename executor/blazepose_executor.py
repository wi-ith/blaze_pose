import tensorflow as tf
from utils import dataloader_utils as dlu
from . import distributed_executor as executor
import numpy as np
import cv2
import sys

class BlazePoseDistributedExecutor(executor.DistributedExecutor):
    def __init__(self, params, model_fn, eval_model_fn, loss_fn=None, eval_fn=None):
        self._params = params
        self._model_fn = model_fn
        self._eval_model_fn = eval_model_fn
        self._loss_fn = loss_fn
        self._eval_fn = eval_fn
        self._summary_writer = tf.summary.create_file_writer(params['train']['model_dir'])

    def post_processing(self, prediction, input_img, heatmaps=None, reg_gt=None):
        pred_sum = tf.expand_dims(tf.reduce_sum(prediction, axis=-1),axis=-1)
        pred_sum = tf.clip_by_value(pred_sum, clip_value_min=0, clip_value_max=1, name=None)
        resized_inputimg = tf.image.resize(input_img,pred_sum.shape[1:3])
        resized_inputimg = tf.clip_by_value((resized_inputimg+1.)*0.5, clip_value_min=0, clip_value_max=1, name=None)
        save_pred = np.broadcast_to(pred_sum,resized_inputimg.shape)
        if heatmaps!=None:
            heatmaps_sum = tf.expand_dims(tf.reduce_sum(heatmaps, axis=-1),axis=-1)
            heatmaps_sum = tf.clip_by_value(heatmaps_sum, clip_value_min=0, clip_value_max=1, name=None)
            heatmaps_sum = np.broadcast_to(heatmaps_sum, resized_inputimg.shape)
            return np.concatenate([np.uint8(resized_inputimg*255), np.uint8(save_pred*255), np.uint8(heatmaps_sum*255)], axis=2)
        else:
            return np.concatenate([np.uint8(resized_inputimg*255), np.uint8(save_pred*255)], axis=2)

    def post_processing_reg(self, prediction, input_img, heatmaps, reg_logit=None, reg_gt=None):

        pred_sum = tf.expand_dims(tf.reduce_sum(prediction, axis=-1), axis=-1)
        pred_sum = tf.clip_by_value(pred_sum, clip_value_min=0, clip_value_max=1, name=None)
        resized_inputimg = tf.image.resize(input_img, pred_sum.shape[1:3])
        resized_inputimg = tf.clip_by_value((resized_inputimg + 1.) * 0.5, clip_value_min=0, clip_value_max=1,
                                            name=None)
        save_pred = np.broadcast_to(pred_sum, resized_inputimg.shape)
        heatmaps_sum = tf.expand_dims(tf.reduce_sum(heatmaps, axis=-1), axis=-1)
        heatmaps_sum = tf.clip_by_value(heatmaps_sum, clip_value_min=0, clip_value_max=1, name=None)
        heatmaps_sum = np.broadcast_to(heatmaps_sum, resized_inputimg.shape)

        point_list = tf.unstack(reg_logit, axis = 0)
        gt_list = tf.unstack(reg_gt, axis = 0)
        pt_output_list = []
        gt_output_list = []
        for batch_pt, gt in zip(point_list,gt_list):
            point_map = np.zeros(resized_inputimg.shape[1:], dtype=np.uint8)
            gt_map = np.zeros(resized_inputimg.shape[1:], dtype=np.uint8)
            pt_list = tf.unstack(batch_pt)
            gt_list = tf.unstack(gt)

            for one_pt, one_gt in zip(pt_list, gt_list):
                output_sized_pt = np.int32([one_pt[0]*pred_sum.shape[1],one_pt[1]*pred_sum.shape[2]])
                point_map = cv2.circle(point_map,output_sized_pt,2,(0,255,255),2)
                output_sized_gt = np.int32([one_gt[0]*pred_sum.shape[1],one_gt[1]*pred_sum.shape[2]])
                gt_map = cv2.circle(gt_map,output_sized_gt,2,(0,255,255),2)
            pt_output_list.append(np.copy(point_map))
            gt_output_list.append(np.copy(gt_map))
        save_ptmap = np.stack(pt_output_list)
        save_gtmap = np.stack(gt_output_list)
        return np.concatenate(
            [np.uint8(resized_inputimg * 255), np.uint8(save_pred * 255), np.uint8(heatmaps_sum * 255), save_ptmap, save_gtmap], axis=2)
        # else:
        #     return np.concatenate([np.uint8(resized_inputimg * 255), np.uint8(save_pred * 255)], axis=2)

        # return pred_sum
        # return np.concatenate([save_img,save_label],axis=2)

    # def post_processing(self, input_img):
    #     save_img = np.array(tf.cast((input_img[:, :, :, :] + 1) * 127.5, dtype=tf.uint8))
    #     save_label = np.array(tf.cast(label[:, :, :], dtype=tf.uint8))
    #     save_pred = np.array(tf.cast(prediction[:, :, :, :], dtype=tf.float32))
    #     ymin = tf.reduce_min(save_pred)
    #     ymax = tf.reduce_max(save_pred - ymin)
    #     depth = (save_pred - ymin) / ymax
    #     depth = np.uint8(depth*255)
    #     depth = np.broadcast_to(depth,save_img.shape)
    #     save_label = np.broadcast_to(np.expand_dims(save_label,axis=-1),save_img.shape)
    #     return np.concatenate([save_img,depth,save_label],axis=2)

    def _get_input_iterator(self, tf_record_pattern):
        dataset = tf.data.Dataset.from_tensor_slices([tf_record_pattern])
        def _parse(x):
            x = tf.data.TFRecordDataset(x)
            return x

        SHUFFLE_BUFFER = 64
        PREFETCH = 256
        dataset = dataset.interleave(_parse, cycle_length=32,
                                     block_length=1,
                                     num_parallel_calls=32,
                                     deterministic=True)
        dataset = dataset.shuffle(SHUFFLE_BUFFER)
        dataset = dataset.map(dlu._parse_function, num_parallel_calls=32)
        final_dataset = dataset.prefetch(buffer_size=1000)
        final_dataset = final_dataset.cache()
        final_dataset = final_dataset.batch(self._params['train']['batch_size'])
        return final_dataset

    def _create_replicated_step(self,
                                model,
                                loss_fn,
                                optimizer
                                ):
        def _replicated_step(inputs, lr):
            images, heatmaps, offsetmaps, regression = dlu.batch_inputs(inputs, params=self._params, istraining=True)
            # images = np.random.rand(16, 256, 256, 3)
            # heatmap = np.random.rand(16, 256, 256, 8)
            # offsetmap = np.random.rand(16, 256, 256, 2, 8)
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                prediction_loss = loss_fn(outputs, heatmaps, offsetmaps, regression)

                if self._params['train']['type']=='regression':
                    grads = tape.gradient(prediction_loss['reg_loss'], model.trainable_variables)
                    optimizer(learning_rate=lr).apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
                else:
                    grads = tape.gradient(prediction_loss['total_loss'], model.trainable_variables)
                    optimizer(learning_rate=lr).apply_gradients(grads_and_vars = zip(grads, model.trainable_variables))
            return prediction_loss, images, outputs, heatmaps, regression

        return _replicated_step


    def _create_train_step(self,
                           model,
                           loss_fn,
                           optimizer,
                           metric=None):

        replicated_step = self._create_replicated_step(model,
                                                       loss_fn,
                                                       optimizer)

        def train_step(iterator, epoch, lr):
            losses={}

            for step, parsed_record in enumerate(iterator):
                losses, input_img, outputs, heatmaps, regression = replicated_step(parsed_record, lr)
                cur_step = (self._params['train']['train_samples']//self._params['train']['batch_size']+1)*epoch + step

                if cur_step % self._params['train']['summary_save_step']==0:
                    self.summary_fn('edl_loss', losses['edl_loss'],cur_step)
                    self.summary_fn('x_loss', losses['x_loss'], cur_step)
                    self.summary_fn('y_loss', losses['y_loss'], cur_step)
                    self.summary_fn('reg_loss', losses['reg_loss'], cur_step)
                    self.summary_fn('learning_rate',lr,cur_step)

                    if self._params['train']['type'] == 'heatmap':
                        self.summary_image_fn(
                            self.post_processing(
                                outputs['heatmap_logit'],
                                input_img,
                                heatmaps
                            ),
                            cur_step,
                            'train_results'
                        )
                    if self._params['train']['type'] == 'regression':
                        self.summary_image_fn(
                            self.post_processing_reg(
                                outputs['heatmap_logit'],
                                input_img,
                                heatmaps,
                                outputs['reg_logit'],
                                regression
                            ),
                            cur_step,
                            'train_reg_results'
                        )
                    print('{} epoch, {} step, total_loss : {}'.format(epoch,cur_step,losses['total_loss']))

            return losses

        return train_step


    def train(self,
              train_input_fn,
              val_input_fn
              ):
        initial_epoch = 0
        total_epochs = self._params['train']['total_epochs']
        model = self.model_fn(self._params)
        if self._params['train']['load_weight_path'] is not None:
            model.load_weights(filepath=self._params['train']['load_weight_path'])
            initial_epoch = int(self._params['train']['load_weight_path'].split('/')[-1].split('-')[-2])
            initial_epoch += 1
            print('training start from ',initial_epoch,' epoch')

        eval_model = self.eval_model_fn(self._params)
        optimizer = model.optimizer
        train_step = self._create_train_step(
            model=model,
            loss_fn=self.loss_fn(),
            optimizer=optimizer)

        _lr = self._params['train']['learning_rate']
        for epoch in range(total_epochs):
            epoch = epoch + initial_epoch
            updated_lr = _lr['init_learning_rate'] * (
                        (_lr['learning_rate_decay_rate']) ** (epoch // _lr['learning_rate_decay_epochs']))
            if self._params['eval']['bento_save'] == '':
                train_step(train_input_fn, epoch, updated_lr)
            ckpt_path = self._params['train']['model_dir'] + "{}-epoch".format(epoch)
            model.save_weights(filepath=ckpt_path)
            if epoch%3==0 and epoch > 0 :
                eval_model.load_weights(filepath=ckpt_path)
                self.eval(val_input_fn, eval_model, epoch)


    def _create_test_step(self,
                          eval_fn,
                          model):

        def _test_step(iterator, epoch):
            mse_fn = tf.keras.metrics.MeanSquaredError()
            total_score = 0
            total_val_number = 0
            total_rmse = 0
            print('start evaluation')
            max_range = int(self._params['eval']['eval_samples']/self._params['eval']['batch_size'])
            random_step = np.random.randint(0, max_range - 1)
            total_num = 0
            collect_sum = 0
            if self._params['train']['type'] == 'regression':
                visibility_sum = 0
                vis_cnt = 0
            output_stride = self._params['architecture']['input_size']['output_stride']
            input_height = self._params['architecture']['input_size']['height']
            input_width = self._params['architecture']['input_size']['width']
            for step, parsed_record in enumerate(iterator):
                if step%100==0:
                    print('eval ',step,' steps')

                num_example = parsed_record['image/encoded'].shape[0]
                total_val_number += num_example
                images, heatmaps, offsetmaps, regression_ev = dlu.batch_inputs(parsed_record, self._params,istraining = False)
                val_logit = model(images, training=False)

                # outputs2, _ = model.predict(self.load_image())
                # self.save_image(outputs2, 'inter')
                if self._params['eval']['bento_save']!='':
                    tf.saved_model.save(model, self._params['eval']['bento_save'])
                    sys.exit(0)
                if random_step == step:
                    save_img = images
                    save_heatmap = heatmaps
                    save_logit = val_logit
                    save_reg = regression_ev

                batch_max_idx = []

                if self._params['train']['type'] == 'heatmap':
                    for batch_ in range(val_logit['heatmap_logit'].shape[0]):
                        one_batch_idx = []
                        for jt_ in range(self._params['architecture']['input_size']['num_keypoints']):
                            raw_idx = np.argmax(val_logit['heatmap_logit'][batch_,:,:,jt_])
                            one_batch_idx.append((int(raw_idx%(input_height/output_stride)),int(raw_idx//(input_width/output_stride))))
                        batch_max_idx.append(np.copy(np.array(one_batch_idx)))
                    batch_max_idx = np.array(batch_max_idx)

                    gt_idx = np.array(regression_ev[:,:,:2]*(input_height/output_stride)).astype(np.int32)
                    torso_diag = np.linalg.norm(batch_max_idx - gt_idx, axis=-1)

                    chk_zero = np.sum(regression_ev[:, :, :2],-1)
                    zero_cnt = np.sum(chk_zero == 0)
                    inner = (input_height/output_stride)*self._params['eval']['pck_rate']
                    mask_col = torso_diag < inner
                    total_num += (mask_col.size - zero_cnt)
                    collect_sum += np.sum(mask_col)

                    one_rmse = mse_fn(y_true = heatmaps, y_pred = val_logit['heatmap_logit'])
                    # one_rmse = np.abs(np.mean(np.square(one_val_depth-one_label))**(0.5))
                    total_rmse += one_rmse
                elif self._params['train']['type'] == 'regression':
                    gt_idx = np.array(regression_ev[:, :, :2] * (input_height)).astype(np.int32)
                    reg_idx = np.array(val_logit['reg_logit'][:,:,:2] * (input_height)).astype(np.int32)

                    gt_vis = np.array(regression_ev[:, :, 2])
                    reg_vis = np.array(val_logit['reg_logit'][:,:, 2])

                    torso_diag_reg = np.linalg.norm(gt_idx - reg_idx, axis=-1)
                    vis_reg = np.average(np.linalg.norm(gt_vis - reg_vis, axis=-1))

                    chk_zero = np.sum(regression_ev[:, :, :2], -1)
                    zero_cnt = np.sum(chk_zero == 0)
                    inner_reg = (input_height) * self._params['eval']['pck_rate']
                    mask_col_reg = torso_diag_reg < inner_reg
                    total_num += (mask_col_reg.size - zero_cnt)
                    collect_sum += np.sum(mask_col_reg)
                    visibility_sum += vis_reg
                    vis_cnt += 1

                    one_rmse = mse_fn(y_true=heatmaps, y_pred=val_logit['heatmap_logit'])
                    # one_rmse = np.abs(np.mean(np.square(one_val_depth-one_label))**(0.5))
                    total_rmse += one_rmse


                # total_score += eval_fn(val_logit,labels)
            cur_step = int((self._params['train']['train_samples']/self._params['train']['batch_size']) * (epoch+1))
            _rmse = total_rmse/total_val_number
            _pck = collect_sum / total_num
            # _score = total_score/(total_val_number/self._params['eval']['batch_size'])
            self.summary_fn('val_mse', _rmse, cur_step)
            self.summary_fn('val_pck', _pck, cur_step)
            if self._params['train']['type'] == 'heatmap':
                self.summary_image_fn(
                    self.post_processing(
                        save_logit['heatmap_logit'],
                        save_img,
                        None
                    ),
                    cur_step,
                    'eval_results'
                )
            if self._params['train']['type'] == 'regression':
                self.summary_fn('val_visibility', visibility_sum/vis_cnt, cur_step)
                self.summary_image_fn(
                    self.post_processing_reg(
                        save_logit['heatmap_logit'],
                        save_img,
                        save_heatmap,
                        save_logit['reg_logit'],
                        save_reg
                    ),
                    cur_step,
                    'eval_reg_results'
                )
            print('{} epoch, {} step, rmse : {}, pck : {}'.format(epoch, cur_step, _rmse, _pck))
        return _test_step


