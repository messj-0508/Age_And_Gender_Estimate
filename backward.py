#coding:utf-8

import os
import tensorflow as tf
import numpy as np

import forward
import generateds

BATCH_SIZE = 16
STEPS = 15000

# 学习率衰减的超参数
LR_BASE = 1e-3 # 学习率的基础值
DECAY_STEPS = 5000 # 每多少轮衰减一次
LR_DECAY = 0.2 # 学习率衰减因子

REGULARIZER = 1e-5
KEEP_PROB = 0.6

MODEL_SAVE_PATH = "./models"
MODEL_NAME = "model_" # model名称前缀
num = 414650


def loss_fn(age_labels, age_logits, gender_labels, gender_logits):
    age_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=age_labels, logits=age_logits)
    age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy)

    gender_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gender_labels,
                                                                          logits=gender_logits)
    gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy)

    # l2 regularization
    total_loss = tf.add_n(
        [gender_cross_entropy_mean, age_cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return total_loss

def backward():
    with tf.Graph().as_default():
        # 导入数据
        images, age_labels, gender_labels = generateds.get_tfRecord(BATCH_SIZE, True)
        train_mode = tf.placeholder(tf.bool)

        # 设置预测值及对应的网络
        age_logits, gender_logits, _ = forward.forward(images,
                                                       keep_probability=KEEP_PROB,
                                                       phase_train=train_mode,
                                                       weight_decay=REGULARIZER)

        # 设置loss函数
        loss = loss_fn(age_labels, age_logits, gender_labels, gender_logits)

        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)


        # 全局当前批训练的轮数
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # 设置学习率衰减
        lr = tf.train.exponential_decay(LR_BASE, global_step, 10000, LR_DECAY, staircase = True)

        # 设置训练计划，优化方式为Adam，
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        # 设置训练的保存器，用于保存模型
        saver = tf.train.Saver(max_to_keep=100)

        with tf.Session() as sess:
            # 变量初始化
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)

            # 断点续训
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("restore and continue training!")

            # 开启输入队列线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # 开始训练
            for i in range(STEPS):
                # 一次训练
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={train_mode: True})
                # 每隔千轮，输出一次loss，便于直接观察模型优化趋势，同时保存一次模型。
                if i % 10 == 0:
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                if i % 1000 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            save_path = saver.save(sess, os.path.join(MODEL_SAVE_PATH, "model.ckpt"), global_step=global_step)
            print("Model saved in file: %s" % save_path)
            coord.request_stop()
            coord.join(threads)

def main():
    # 执行训练函数（反向传播中包含前向传播，因此是完整的训练过程）
    backward()

# 当该模块为运行的主模块，执行下列语句
if __name__ == '__main__':
    main()


