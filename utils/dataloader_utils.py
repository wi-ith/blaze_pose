import tensorflow as tf
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2

def _parse_function(example_proto):
    keys_to_features = {
        'image/height':
            tf.io.FixedLenFeature((), tf.int64),
        'image/width':
            tf.io.FixedLenFeature((), tf.int64),
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/keypoints':
            tf.io.FixedLenSequenceFeature((), tf.int64, allow_missing=True, default_value=[0]),
    }
    return tf.io.parse_single_example(example_proto, keys_to_features)

def makeGaussian2(x_center=0, y_center=0, theta=0, sigma_x = 1, sigma_y=1, x_size=640, y_size=480):
    # x_center and y_center will be the center of the gaussian, theta will be the rotation angle
    # sigma_x and sigma_y will be the stdevs in the x and y axis before rotation
    # x_size and y_size give the size of the frame

    theta = 2*np.pi*theta/360
    x = np.arange(0,x_size, 1, float)
    y = np.arange(0,y_size, 1, float)
    y = y[:,np.newaxis]
    sx = sigma_x
    sy = sigma_y
    x0 = x_center
    y0 = y_center

    # rotation
    a=np.cos(theta)*x -np.sin(theta)*y
    b=np.sin(theta)*x +np.cos(theta)*y
    a0=np.cos(theta)*x0 -np.sin(theta)*y0
    b0=np.sin(theta)*x0 +np.cos(theta)*y0

    return np.exp(-(((a-a0)**2)/(2*(sx**2)) + ((b-b0)**2) /(2*(sy**2))))


def decode_jpeg(image_buffer, channels, scope=None):
    with tf.name_scope(name=scope):
        image = tf.image.decode_image(image_buffer, channels)
        return image

# def inputs(parsed_tfr, params):
#     # with tf.device('/cpu:0'):
#     images_batch, boxes_batch = batch_inputs(parsed_tfr, params)
#
#     return images_batch, boxes_batch
def augmentation_():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    aug_seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        # sometimes(iaa.CropAndPad(
        #                 percent=(-0.05, 0.1),
        #                 pad_mode=["constant", "maximum", "mean", "median","minimum"],
        #                 pad_cval=(0, 255)
        #             )),
        # sometimes(iaa.Affine(
        #     scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # scale images to 80-120% of their size, individually per axis
        #     translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
        #     rotate=(-5, 5),  # rotate by -45 to +45 degrees
        #     shear=(-5, 5),  # shear by -16 to +16 degrees
        #     order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        #     cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        #     mode=["constant"]  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        # )),
        # iaa.GaussianBlur(sigma=(0, 3.0))
    ])
    return aug_seq

def batch_inputs(parsed_record, params, istraining=True):
    image_batch = []
    heatmap_batch = []
    offsetmap_batch = []
    regression_batch = []
    keypoints_list = parsed_record['image/keypoints']
    key_arr = np.array(keypoints_list)

    # reshaped_key = np.reshape(key_arr, [key_arr.shape[0], -1, 4])
    # keypt_coordi = np.copy(reshaped_key[:, :, :2])
    # keypt_occ = np.copy(reshaped_key[:, :, 2:])
    reshaped_key = np.reshape(key_arr, [key_arr.shape[0], -1, 3])
    keypt_coordi = np.copy(reshaped_key[:, :, :2])
    visibility = np.copy(reshaped_key[:, :, 2])

    input_height = params['architecture']['input_size']['height']
    input_width = params['architecture']['input_size']['width']
    output_stride = params['architecture']['input_size']['output_stride']

    output_height = int(input_height / output_stride)
    output_width = int(input_width / output_stride)
    sigma = np.sqrt(output_height * output_width) * params['train']['sigma_k']



    # occluded = np.expand_dims(1-np.max(np.int32((reshaped_key==-1)),axis=-1),axis=-1)
    # ann_vis = np.concatenate([reshaped_key,occluded],axis=-1)

    # if np.sum(occluded)<np.size(occluded):
    #     break

    for q in range(len(parsed_record['image/encoded'])):
        byte_image_jpeg = decode_jpeg(parsed_record['image/encoded'][q], 3, scope='decode_mask_jpeg')
        byte_image_jpeg = tf.expand_dims(byte_image_jpeg, axis=0)
        byte_image_jpeg = np.array(byte_image_jpeg)
        _, height, width, _ = byte_image_jpeg.shape
        expand_key = np.expand_dims(keypt_coordi[q], axis=0)
        if istraining :
            images_aug, points_aug = augmentation_()(images=byte_image_jpeg, keypoints=expand_key)
            # images_aug = byte_image_jpeg
            # points_aug = expand_key
        else:
            images_aug = byte_image_jpeg
            points_aug = expand_key
        heatmap_list = []
        offsetmap_list = []
        scale_pt_x = points_aug[0,:, 0] / width
        scale_pt_y = points_aug[0,:, 1] / height
        scale_pt = np.stack([scale_pt_x,scale_pt_y,visibility[q]/2.],axis=-1)

        regression_batch.append(np.copy(scale_pt))

        # for point, occ_ in zip(points_aug[0], keypt_occ[q]):
        for point in points_aug[0]:
            # if occ_[0] == 0:
            #     heatmap_list.append(np.zeros([output_height, output_width, 1]))
            #     offsetmap_list.append(np.zeros([output_height, output_width, 2]))
            # else:

                h_rate = (output_height / output_stride) / height
                w_rate = (output_width / output_stride) / width

                offset_y = point[1] * h_rate - int(point[1] * h_rate) - 0.5
                offset_x = point[0] * w_rate - int(point[0] * w_rate) - 0.5

                resized_point = (np.minimum((point[0] / width) * output_width,output_width-1), np.minimum((point[1] / height) * output_height,output_height-1))
                resized_point = np.int32(np.array(resized_point))

                offset_map = np.zeros([output_height, output_width, 2])
                offset_map[resized_point[1], resized_point[0], :] = np.array([offset_y, offset_x])

                gaus = makeGaussian2(x_center=resized_point[0], y_center=resized_point[1], theta=0, sigma_x=sigma,
                                     sigma_y=sigma, x_size=output_width, y_size=output_height)
                gaus = tf.expand_dims(gaus, axis=-1)
                # gaus = tf.image.resize(gaus,
                #                        tf.stack([output_height,
                #                                  output_width]))
                heatmap_list.append(gaus)
                offsetmap_list.append(offset_map)
        image_resize = tf.image.resize(images_aug,
                                       tf.stack([input_height,
                                                 input_width]))

        heatmap_stack = tf.squeeze(tf.stack(heatmap_list, axis=-1))
        offsetmap_stack = tf.squeeze(tf.stack(offsetmap_list, axis=-1))
        image_batch.append(tf.squeeze(image_resize))
        heatmap_batch.append(heatmap_stack)
        offsetmap_batch.append(offsetmap_stack)

    stacked_images = (tf.stack(image_batch, axis=0)/255.-0.5)*2
    stacked_heatmaps = tf.stack(heatmap_batch, axis=0)
    offsetmap_batch = tf.stack(offsetmap_batch, axis=0)
    regression_batch = tf.stack(regression_batch, axis=0)

    return stacked_images, stacked_heatmaps, offsetmap_batch, regression_batch
    # for q in range(len(parsed_tfr['image/encoded'])):
    #     byte_image_jpeg = decode_jpeg(parsed_record['image/encoded'][q], 3, scope='decode_mask_jpeg')
    #     byte_image_jpeg = tf.expand_dims(byte_image_jpeg,axis=0)
    #     byte_image_jpeg = tf.convert_to_tensor(byte_image_jpeg)
    #     _, height,width,_ = byte_image_jpeg.shape
    #     expand_key = tf.expand_dims(reshaped_key[q],axis=0)
    #     images_aug, points_aug = aug_seq(images=byte_image_jpeg, keypoints=expand_key)
    #
    #     heatmap_list = []
    #     for point in points_aug[0]:
    #         gaus = makeGaussian2(x_center=point[0], y_center=point[1], theta=0, sigma_x=10,
    #                              sigma_y=10, x_size=width, y_size=height)
    #         gaus = tf.expand_dims(gaus,axis=-1)
    #         gaus = tf.image.resize(gaus,
    #                                tf.stack([params['architecture']['input_size']['height'],
    #                                          params['architecture']['input_size']['width']]))
    #         heatmap_list.append(gaus)
    #     image_resize = tf.image.resize(images_aug,
    #                                    tf.stack([params['architecture']['input_size']['height'],
    #                                              params['architecture']['input_size']['width']]))
    #     heatmap_stack = tf.squeeze(tf.stack(heatmap_list,axis = -1))
    #     image_batch.append(tf.squeeze(image_resize))
    #     points_batch.append(heatmap_stack)
    #
    #
    # stacked_iamges = tf.stack(image_batch,axis=0)
    # stacked_heatmaps = tf.stack(points_batch, axis=0)
    #
    # return stacked_iamges, stacked_heatmaps

