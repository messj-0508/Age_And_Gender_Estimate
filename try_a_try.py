# coding: utf-8

import cv2
import tensorflow as tf
import numpy as np
import forward
import dlib
from imutils.face_utils import FaceAligner

'''
绘图方法——添加标签
image：输入图像
point：起点坐标
label：标签
font：字体格式
font_scale：字体格式
thickness：画线粗细
'''
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x,y = point
    cv2.rectangle(image, (x, y-size[1]), (x+size[0], y), (255,0,0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255,255,255), thickness)

def main(sess, age, gender, train_mode, img_input):
    # 使用dlib自带的frontal_face_detector作为我们的人脸提取器,输出人脸个数
    detector = dlib.get_frontal_face_detector()

    # 使用dlib提供的模型构建特征提取器,输出人脸关键点
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # 使用imutil提供的脸部校准
    fa = FaceAligner(predictor, desiredFaceWidth=160)

    # 图片大小设置
    img_size = 160

    # 使用cv2调用摄像头

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    while True:
        # 获取摄像头的一帧画面
        ret, img = cap.read()

        if not ret:
            print("error:未捕获画面")
            return -1;

        # 图片通道转换
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w, _ = np.shape(input_img)

        # 获取图片中脸的位置
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        # 绘制方框指出脸的位置
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right()+1, d.bottom()+1, d.width(), d.height()

            # 计算出对角线的两个顶点坐标
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
            yw2 = min(int(y2 + 0.4 * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 脸部校正
            faces[i, :, :, :] = fa.align(input_img, gray, detected[i])

        # 预测年龄与性别
        if len(detected) > 0:
            ages,genders = sess.run([age, gender], feed_dict={img_input: faces, train_mode: False})

        # 绘制标签
        for i, d in enumerate(detected):
            label = "{}, {}".format(int(ages[i]), "F" if genders[i] == 0 else "M")
            draw_label(img, (d.left(), d.top()), label)

        # 展示图片
        cv2.imshow("result", img)

        # 关停指令
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

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

if __name__ == '__main__':
    model_path = "./models"
    sess, age, gender, train_mode, img_input = load_model(model_path)
    main(sess, age, gender, train_mode, img_input)
