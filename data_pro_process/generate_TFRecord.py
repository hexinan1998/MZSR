# path of label D:\dataset\DIV2K\DIV2K_train_HR
# path of img D:\dataset\DIV2K\DIV2K_train_LR_bicubic\X2

import imageio
import os
import glob
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from matplotlib import pyplot as plt
'''
@path 存储图片的路径

return  数字化图像的一个 imageio 对象
'''
def imread(path):
    img = imageio.imread(path)
    return img

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

'''
@imgs 输入的图
@modulo  以什么为模


return  可以整除模的剪裁
'''
def modcrop(imgs, modulo):
    imgs_shape = imgs.shape
    imgs_shape = np.asarray(imgs_shape)
    if len(imgs_shape):
        imgs_shape = imgs_shape - imgs_shape%modulo
        out = imgs[:imgs_shape[0],:imgs_shape[1]]
    elif len(imgs_shape):
        imgs_shape = imgs_shape[0:2]
        imgs_shape = imgs_shape - imgs_shape%modulo
        out = imgs[:imgs_shape[0],:imgs_shape[1],:]
    return out

def write_to_tfrecord(write, label, image):
    feature = {
        "image":tf.train.Feature(bytes_list=tf.train.BytesList()),
        "label":tf.train.Feature(bytes_list=tf.train.BytesList())
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    write.write(example.SerializeToString())
    return

def generate_TFRecord(data_path,label_path,tfrecord_file,patch_h,patch_w,stride):
    label_list = np.sort(np.asarray(glob.glob(label_path)))
    img_list = np.sort(np.asarray(glob.glob(data_path)))

    offset = 0

    fileNum = len(label_list)

    patches=[]
    labels=[]

    for n in range(fileNum):
        print(f'[*] Image number {n}/{fileNum}')
        img = imread(img_list[n])
        label = imread(label_list[n])
        # 这个断言 表示取出的 图片 的标号是一样的
        assert os.path.basename(img_list[n])[:-6] == os.path.basename(label_list[n])[:-4]

        img = modcrop(img,scale)
        label = modcrop(label,scale)

        x, y, ch = label.shape

        for i in range(0+offset,x-patch_h+1,stride):
            for j in range(0+offset, y-patch_w+1,stride):
                patch_d = img[i // scale:i // scale + patch_h // scale, j // scale:j // scale + patch_w //scale]
                patch_l = img[i:i + patch_h,j: j + patch_w]
                
                if np.log(gradients(patch_l.astype(np.float64)/255.)+1e-10) >= -6.0:
                    patches.append(patch_d.tobytes())
                    labels.append(patch_l.tobytes())

        np.random.seed(36)
        np.random.shuffle(patches)
        np.random.seed(36)
        np.random.shuffle(labels)
        print(f"Num of patches:{len(patches)}")
        print(f"shape:{patch_h},{patch_h},{ch}")
        
        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            for i in range(len(patches)):
                write_to_tfrecord(writer, labels[i], patches[i])

def img_show(img):
    fig = plt.figure()
    img_show = fig.add_subplot(111)
    img_show.imshow(img)
    plt.show()
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--scale' , dest="scale",help='Scaling Factor for Super-Resolution',type=int, default=2)
    parser.add_argument('--labelpath', dest='labelpath', help='Path to HR images (./DIV2K_train_HR)', default="D:\dataset\DIV2K\DIV2K_train_HR")
    parser.add_argument('--datapath', dest='datapath', help='Path to LR images (./DIV2K_train_LR_bicubic)', default="D:\dataset\DIV2K\DIV2K_train_LR_bicubic\X2")
    parser.add_argument('--tfrecord', dest='tfrecord', help='Save path for tfrecord file', default='large_dataset/train_SR_X2')

    options = parser.parse_args()

    scale = options.scale
    labelpath = os.path.join(options.labelpath,"*.png")
    datapath = os.path.join(options.datapath,"*.png")

    tfrecord_file = options.tfrecord + '.tfrecord'

    generate_TFRecord(datapath, labelpath, tfrecord_file,48*scale,48*scale,120)
    
    # c = imread("C://Users//Hexinan_cp//Desktop//1.jpg")
    # c = modcrop(c,2)
    # print(type(c))
