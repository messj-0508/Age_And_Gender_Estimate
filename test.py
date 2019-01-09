#coding: utf-8

import time
import tensorflow as tf
import numpy as np
import forward
import backward
import generateds

# 每轮test之间的停顿间隙
#TEST_INTERVAL_SECS = 20
BATCH_SIZE = 16
    

def test():
    '''
    tf.Graph().as_default():一个将某图设置为默认图，并返回一个上下文的管理器。如果不显式添加一个默认图，系统会自动设置一个全局的默认图。所设置的默认图，在模块范围内所定义的节点都将默认加入默认图中。
    '''
    with tf.Graph().as_default() as g:

        # 导入数据,num_epochs=1即测试集仅输出一轮(epoch)
        images, age_labels, gender_labels = generateds.get_tfRecord(BATCH_SIZE, True, num_epochs = 1)
        train_mode = tf.placeholder(tf.bool)

        # 设置预测值及对应的网络
        age_logits, gender_logits, _ = forward.forward(images,
                                                       keep_probability=backward.KEEP_PROB,
                                                       phase_train=train_mode,
                                                       weight_decay=backward.REGULARIZER)
        
        '''
        tf.euqal:相当于tensor变量比较的“==”符号。
        tf.cast(x,dtype,name=None):将x的数据格式转化成dtype.
        '''
        # 计算年龄的平均绝对误差
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        prob_age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        abs_age_error = tf.losses.absolute_difference(prob_age, age_labels)
        
        # 计算性别的准确率
        gender_current_prediction = tf.equal(tf.argmax(gender_logits,1),gender_labels)
        gender_accuracy = tf.reduce_mean(tf.cast(gender_current_prediction, tf.float32))

        # 设置训练的保存器
        saver = tf.train.Saver()

        # 每个batch的评价指标汇总
        mean_error_age, mean_gender_acc = [], []
        '''
        TEST
        '''
        with tf.Session() as sess:

            init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            sess.run(init_op)

            # 加载模型（参数）
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 获取该模型的全局批训练的轮数
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                try:
                    while not coord.should_stop():
                        real_gender, real_age, image_val, gender_acc_val, abs_age_error_val = sess.run(
                            [gender_labels, age_labels, images, gender_accuracy, abs_age_error], {train_mode: False})
                        mean_error_age.append(abs_age_error_val)
                        mean_gender_acc.append(gender_acc_val)
                except tf.errors.OutOfRangeError:
                    print('Done!')
                finally:
                    coord.request_stop()
                coord.join(threads)
            else:
                print("No CheckPoint File Found!")
                return 
        #time.sleep(TEST_INTERVAL_SECS)
        print("After %s training step(s), test accuracy of gender = %g, test MAE of age = %g" % (global_step, np.mean(mean_gender_acc), np.mean(mean_error_age)))

def main():
    test()

# 当该模块为运行的主模块，执行下列语句
if __name__ == '__main__':
    main()

