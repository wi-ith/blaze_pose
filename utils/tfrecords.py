import io
import hashlib
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import json
import numpy as np
from PIL import Image
# from utils.superbai \
import utils_superb as du
import cv2
import argparse

parser = argparse.ArgumentParser(description='',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-ir', '--image_root', type=str, default=None,
                    help='')
parser.add_argument('-lr', '--label_root', type=str, default=None,
                    help='')
parser.add_argument('-of', '--outout_folder', type=str, default=None,
                    help='')

parser.add_argument('--port', type=str, default=None,
                    help='')
parser.add_argument('--mode', type=str, default=None,
                    help='')

args = parser.parse_args()


_image_root = args.image_root
_label_root = args.label_root
outout_folder = args.outout_folder
file_list = os.listdir(_image_root)
writer_train = tf.python_io.TFRecordWriter(outout_folder + 'train.record')
writer_val = tf.python_io.TFRecordWriter(outout_folder + 'val.record')
test_mode = False
total_image_num=0
wrong_foots_cnt=0

# for l, json_file_name in enumerate(json_file_list):
val_cnt=0
train_cnt=0
for l, image_name in enumerate(file_list):
    if l%500==0:
        print(l,' / ',len(file_list),' done.')
    label_path = _label_root + image_name.split('.')[0]+'.json'
    foot_cnt = 0
    with open(label_path, 'r', encoding='latin1') as label_file:
        label_data = json.load(label_file)
        annot = []
        full_path = _image_root + image_name
        try:
            with tf.gfile.GFile(full_path, 'rb') as fid:
                encoded_image = fid.read()
        except:
            continue
        encoded_image_io = io.BytesIO(encoded_image)
        image = Image.open(encoded_image_io)
        width, height = image.size

        for poly_ in label_data["objects"]:
            if poly_['annotation_type'] == 'box':
                # print(label_path)
                try:
                    class_name = poly_["className"].lower()
                except:
                    class_name = poly_["class_name"].lower()
                # if class_name.lower() == "left_foot":

                # try :
                coordi = poly_["annotation"]["coord"]
                offset_h = coordi["height"] * 0.05
                offset_w = coordi["width"] * 0.05
                xmin = np.maximum(coordi["x"] - offset_w, 0)
                ymin = np.maximum(coordi["y"] - offset_h, 0)
                xmax = np.minimum(coordi["x"] + coordi["width"] + offset_w, width - 1)
                ymax = np.minimum(coordi["y"] + coordi["height"] + offset_h, height - 1)
                cbox = (xmin, ymin, xmax, ymax)
                cropped_img = image.crop(cbox)
                img_byte_arr = io.BytesIO()
                cropped_img.save(img_byte_arr, format='JPEG')
                cropped_width, cropped_height = cropped_img.size
                cropped_encoded_image = img_byte_arr.getvalue()
                if test_mode:
                    encoded_img = np.fromstring(cropped_encoded_image, dtype=np.uint8)
                    img_cv = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

                keypt_list = []

                for poly_ in label_data["objects"]:
                    if poly_['annotation_type'] == 'keypoint':
                        key_coordi = []
                        coordi = poly_["annotation"]["coord"]["points"]
                        for point in coordi:
                            if point['state']['valid']:
                                key_coordi.extend([int(point['x']), int(point['y']), int(point['state']['visible'])])
                            else:
                                key_coordi.extend([0, 0, 0])

                        keypt_list.append(np.array(key_coordi))

                ## pick the keypoint which is in bbox
                inbox_kpt = np.array([])

                for kpt in keypt_list:
                    reshaped_kpt = np.reshape(kpt, [-1, 3])
                    reshaped_kpt = reshaped_kpt[:, :2]
                    nonzero_kpt = []
                    for one_k in range(reshaped_kpt.shape[0]):
                        if not 0 in list(reshaped_kpt[one_k, :]):
                            nonzero_kpt.extend(list(reshaped_kpt[one_k, :]))
                    nonzero_kpt = np.reshape(np.array(nonzero_kpt), [-1, 2])
                    check_x = np.logical_and(nonzero_kpt[:, 0] < xmax, nonzero_kpt[:, 0] > xmin)
                    check_y = np.logical_and(nonzero_kpt[:, 1] < ymax, nonzero_kpt[:, 1] > ymin)

                    if np.sum(check_x) == len(check_x) and np.sum(check_y) == len(check_y):
                        kpt = np.reshape(kpt, [-1, 3])
                        kpt_x = np.maximum(kpt[:, 0] - xmin, 0).astype(np.int64)
                        kpt_y = np.maximum(kpt[:, 1] - ymin, 0).astype(np.int64)
                        kpt_vis = kpt[:, 2].astype(int)

                        kpt = np.stack([kpt_x, kpt_y, kpt_vis], axis=-1)
                        kpt = np.reshape(kpt, [-1])
                        if test_mode:
                            for one_pt in list(kpt):
                                cv2.circle(img_cv, np.int32(one_pt[:2]), 3, (255, 0, 255), 3)

                            cv2.imwrite('/home/kdg/dev/tmp/test_pose/' + str(xmin) + image_name, img_cv)
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'image/height': du.int64_feature(cropped_height),
                            'image/width': du.int64_feature(cropped_width),
                            'image/filename': du.bytes_feature(image_name.encode('utf8')),
                            'image/encoded': du.bytes_feature(cropped_encoded_image),
                            'image/keypoints': du.int64_list_feature(kpt),
                        }))
                        if l < len(file_list) * 0.98:
                            writer_train.write(example.SerializeToString())
                            train_cnt += 1
                        else:
                            writer_val.write(example.SerializeToString())
                            val_cnt += 1
                        break

                    #### check if kpts are in box then pick the right one
                # annot.extend(['2',str(round(coordi["x"])),str(round(coordi["y"])),str(round(coordi["x"]+coordi["width"])),str(round(coordi["y"]+coordi["height"]))])
                foot_cnt += 1


writer_train.close()
writer_val.close()
print('wrong foots : ', wrong_foots_cnt)
print('train count : ',train_cnt)
print('val count : ',val_cnt)