# coding: utf-8

import os
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np

import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import dlib

from datetime import datetime
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASE_DIR = "./imdb_crop"
MAT_PATH = "./imdb_crop/imdb.mat"
DB_NAME = "imdb"
tfRecord_train = './data/train.tfrecords'
tfRecord_test = './data/test.tfrecords'
data_path = './data'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfRecord(dataset, tfRecordName):
    # 获取标签信息
    file_name = dataset.file_name
    genders = dataset.gender
    ages = dataset.age
    face_score = dataset.score
    second_face_score = dataset.second_score

    # 获取标签数量
    num_examples = dataset.shape[0]
    print("The number of "+str(tfRecordName)+":"+str(num_examples))

    # 借助脸部校正工具调整数据集
    shape_predictor = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    fa = FaceAligner(predictor, desiredFaceWidth=160)

    # 开始写入
    print('Begin to write', tfRecordName)
    writer = tf.python_io.TFRecordWriter(tfRecordName)

    error=0
    num_pic=0

    # 通过tqdm导入进程条
    for index in tqdm(range(num_examples)):

        # 如果评分过低，则不适合作为数据
        if face_score[index] < 0.75:
            continue
        # 如果年龄不在（0,100）区间内，则不适合作为数据
        if ~(0 <= ages[index] <= 100):
            continue
        # 如果性别未标注，则不适合作为数据
        if np.isnan(genders[index]):
            continue

        # 图片预处理
        try:
            # 读取图片
            image = cv2.imread(os.path.join(BASE_DIR, str(file_name[index][0])), cv2.IMREAD_COLOR)
            # image = imutils.resize(image, width=256)
            # 转为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 检测脸部位置
            rects = detector(gray, 2)
            # 若未检测到人脸，跳过；否则，选取人脸部分作为数据
            if len(rects) != 1:
                continue
            else:
                image_raw = fa.align(image, gray, rects[0])
                image_raw = image_raw.tostring()
        except IOError:  # some files seem not exist in face_data dir
            error = error + 1
            pass
        # 存储格式
        example = tf.train.Example(features=tf.train.Features(feature={
            'age': _int64_feature(int(ages[index])),
            'gender': _int64_feature(int(genders[index])),
            'image_raw': _bytes_feature(image_raw)}))
        # 存储
        writer.write(example.SerializeToString())
        num_pic = num_pic+1
    print("There are ",error," missing pictures" )
    print("Found" ,num_pic, "valid faces")
    writer.close()

# 根据拍照时间与出生年龄计算年龄
def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1

def get_dataset():
    # 读取mat文件
    meta = loadmat(MAT_PATH)

    # 文件路径
    full_path = meta[DB_NAME][0, 0]["full_path"][0]
    # 生日
    dob = meta[DB_NAME][0, 0]["dob"][0]
    # 性别
    gender = meta[DB_NAME][0, 0]["gender"][0]
    # 拍照年份
    photo_taken = meta[DB_NAME][0, 0]["photo_taken"][0]
    # 脸部得分：根据拍照的效果而定
    face_score = meta[DB_NAME][0, 0]["face_score"][0]
    # 脸部第二得分：根据拍照的效果而定
    second_face_score = meta[DB_NAME][0, 0]["second_face_score"][0]
    # 年龄
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    # 合成数据集标签
    data = {"file_name": full_path, "gender": gender, "age": age, "score": face_score,
            "second_score": second_face_score}

    # 制作数据集
    dataset = pd.DataFrame(data)
    return dataset

def generate_tfRecord():
    # 获取数据集
    dataset = get_dataset()

    # 拆分为训练集与测试集
    train_sets,test_sets = train_test_split(dataset,train_size=0.001,random_state=2017)

    # 重置索引
    train_sets.reset_index(drop=True, inplace=True)
    test_sets.reset_index(drop=True, inplace=True)

    # 转化为tfrecord文件
    write_tfRecord(train_sets, tfRecord_train)
    #write_tfRecord(train_sets, tfRecord_test)



def read_tfRecord(tfRecord_path, num_epochs):
    filename_queue = tf.train.string_input_producer([tfRecord_path], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features = {
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'age': tf.FixedLenFeature([], tf.int64),
                    'gender': tf.FixedLenFeature([], tf.int64),
                }
            )

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([160 * 160 * 3])
    image = tf.reshape(image, [160, 160, 3])
    image = tf.reverse_v2(image, [-1])
    image = tf.image.per_image_standardization(image)

    age = features['age']
    gender = features['gender']
    return image, age, gender


def get_tfRecord(batch_size, isTrain=True, num_epochs = None):
    # 根据状态选择数据路径
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    # 读取数据
    image, age, gender =  read_tfRecord(tfRecord_path, num_epochs)
    '''
    name:tf.train.shuffle_batch
    return:  a list of tensors with the same number and types as tensor_list.     
    '''
    img_batch, age_batch, gender_batch = tf.train.shuffle_batch(
            [image, age, gender],
            batch_size = batch_size, # 批处理规模
            num_threads = 2, # 线程数
            capacity = 1300 , # 队列中元素的最大数目
            min_after_dequeue = 1000,) # 出队后队列元素的最小数目
    return img_batch, age_batch, gender_batch


def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()