# coding: utf-8


import cv2
import tensorflow as tf
import numpy as np
import forward
import dlib
from imutils.face_utils import FaceAligner

# import time


class Application():

    def __init__(self, model_path):
        # 开启会话
        sess = tf.Session()

        # 输入
        img_input = tf.placeholder(tf.float32, shape = [None, 160, 160, 3], name='imput_image')
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
            #print("restore model!")
            
        # 使用dlib自带的frontal_face_detector作为我们的人脸提取器,输出人脸个数
        detector = dlib.get_frontal_face_detector()

        # 使用dlib提供的模型构建特征提取器,输出人脸关键点
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # 使用imutil提供的脸部校准
        fa = FaceAligner(predictor, desiredFaceWidth=160)

        self.sess = sess
        self.age = age
        self.gender = gender
        self.train_mode = train_mode
        self.img_input = img_input
        self.detector = detector
        self.fa = fa
        self.img_size = 160

    def predict(self, img_path):
        img = cv2.imread(img_path)

        # 图片通道转换
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_h, img_w, _ = np.shape(input_img)

        # 获取图片中脸的位置
        detected = self.detector(input_img, 1)

        # 预测年龄与性别
        global ages
        global genders
        try:
            if len(detected) > 0:
                faces = np.empty((len(detected), self.img_size, self.img_size, 3))
                
                # 脸部校正
                faces[0, :, :, :] = self.fa.align(input_img, gray, detected[0])
                ages, genders = self.sess.run([self.age, self.gender], feed_dict={self.img_input: faces, self.train_mode: False})          
                label = {'ages': int(ages[0]), 'gender': 'Male' if genders[0] else 'Female'}      
                return label
            else:
                return "-1"
        except UnboundLocalError:
            return "-1"


# 主函数
def main():
    model_path = "./models"
    img_path = "./test.jpg"
    #start = time.time()
    app = Application(model_path)
    print("The initialization of Application speeds %.0f ms"%((time.time()-start)*1000))
    #start = time.time()
    print(app.predict(img_path))
    #print("The predict of Application speeds %.0f ms"%((time.time()-start)*1000))

# 当该模块为运行的主模块，执行下列语句
if __name__ == '__main__':
    main()
