# coding: utf-8


import cv2
import tensorflow as tf
import numpy as np
import forward
import dlib
from imutils.face_utils import FaceAligner

def predict(img_path, sess, age, gender, train_mode, img_input):
    # 使用dlib自带的frontal_face_detector作为我们的人脸提取器,输出人脸个数
    detector = dlib.get_frontal_face_detector()

    # 使用dlib提供的模型构建特征提取器,输出人脸关键点
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # 使用imutil提供的脸部校准
    fa = FaceAligner(predictor, desiredFaceWidth=160)

    # 图片大小设置
    img_size = 160

    img = cv2.imread(img_path)

    # 图片通道转换
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w, _ = np.shape(input_img)

    # 获取图片中脸的位置
    detected = detector(input_img, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))

    # 绘制方框指出脸的位置
    for i, d in enumerate(detected):
        # 脸部校正
        faces[i, :, :, :] = fa.align(input_img, gray, detected[i])

    # 预测年龄与性别
    if len(detected) > 0:
        ages,genders = sess.run([age, gender], feed_dict={img_input: faces, train_mode: False})
    label = "{}, {}".format(int(ages[0]), "F" if genders[0] == 0 else "M")
    return label



# 导入网络与模型
def load_model(model_path):
    # 开启会话
    sess = tf.Session()

    img_input = tf.placeholder(tf.float32, shape = [None, 160, 160, 3], name='imput_image')

    # 标准化
    images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), img_input)
    train_mode = tf.placeholder(tf.bool)

    # 导入网络
    age_logits, gender_logits, _ = forward.forward(images_norm, keep_probability=0.8,
                                                                 phase_train=train_mode,
                                                                 weight_decay=1e-5)

    gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)

    # 初始化
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    # 导入模型
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore model!")
    else:
        pass
    return sess, age, gender, train_mode, img_input


# API接口
def application():
    model_path = "./models"
    sess, age, gender, train_mode, img_input = load_model(model_path)
    
    # 测试次数
    testNum = input("Input the number of test: ")
    for i in range(int(testNum)):
        img_path = input("The path of test picture: ")
        preValue = predict(img_path, sess, age, gender, train_mode, img_input)
        print (preValue)

# 主函数
def main():
    application()

# 当该模块为运行的主模块，执行下列语句
if __name__ == '__main__':
    main()
