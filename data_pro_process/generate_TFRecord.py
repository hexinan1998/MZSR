import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
import os


def decode_img(filename):
    img_string = tf.io.read_file(filename)
    img_decode = tf.io.decode_jpeg(img_string)
    return img_decode

def generate_TFRecord(data_path,label_path,tfrecord_file,patch_h,patch_w,stride):

    pass



if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--scale", dest="scale", help="SR比例因子", type=int, default=2)
    parser.add_argument("--labelpath", dest="labelpath", help="标签的路径")
    parser.add_argument("--datapath", dest="datapath", help="数据的路径")
    parser.add_argument('--tfrecord', dest='tfrecord', help='生成数据保存的路径', default='train_SR_X2')
    options = parser.parse_args()

    scale = options.scale
    labelpath = options.labelpath
    datapath = options.datapath
    tfrecord_file = options.tfrecord + '.tfrecord'

    generate_TFRecord(datapath, labelpath, tfrecord_file, 48 * scale, 48 * scale, 120)

    print("Done")
