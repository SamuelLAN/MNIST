#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import base
import input_data
import tensorflow as tf

'''
 简单的 BP
 这里只有一层隐藏层
 准确率：
    test set accuracy: 94.480002%
'''
class BP(base.NN):
    BASE_LEARNING_RATE = 0.01
    NUM_CLASSES = 10
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
    SHAPE_LIST = [(IMAGE_PIXELS, 1024), (1024, 784), (784, NUM_CLASSES)]
    MODEL_NAME = 'bp_simple'
    REGULAR_BETA = 0.01
    BATCH_SIZE = 128
    DROPOUT_LIST = [0.5, 0.5]


    ''' 自定义 初始化变量 过程 '''
    def init(self):
        # 加载数据
        self.load()

        # 常量
        self.__iterPerEpoch = int(self.__trainSize // self.BATCH_SIZE)
        self.__steps = self.EPOCH_TIMES * self.__iterPerEpoch

        # 输入 与 label
        self.__X = tf.placeholder('float', [None, self.SHAPE_LIST[0][0]])
        self.__y = tf.placeholder('float', [None, self.SHAPE_LIST[-1][-1]])

        # 随训练次数增多而衰减的学习率
        self.__learningRate = self.getLearningRate(
            self.BASE_LEARNING_RATE, self.globalStep, self.BATCH_SIZE, self.__steps, self.DECAY_RATE
        )


    ''' 加载数据 '''
    def load(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.__trainSet = mnist.train
        self.__valSet = mnist.validation
        self.__testSet = mnist.test

        self.__trainSize = self.__trainSet.images.shape[0]
        self.__valSize = self.__valSet.images.shape[0]
        self.__testSize = self.__testSet.images.shape[0]


    ''' 模型 '''
    def model(self):
        self.__output = self.fullConnectModel(self.__X, self.SHAPE_LIST)


    ''' 前向推导 '''
    def inference(self):
        self.__predict = self.fullConnectModel(self.__X, self.SHAPE_LIST, False)


    ''' 计算 loss '''
    def getLoss(self):
        self.__loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.__output, labels=self.__y)
        )


    ''' 计算准确率 '''
    def __getAccuracy(self, batch_x, batch_y):
        labels = tf.argmax(batch_y, 1)
        outputs = tf.argmax(self.__predict, 1)
        correct = tf.equal(labels, outputs)     # 返回 outputs 与 labels 相匹配的结果

        size = self.sess.run(labels).shape[0]   # 计算准确率
        accuracy = tf.divide(tf.reduce_sum(tf.cast(correct, tf.float32)), size)

        feed_dict = {self.__X: batch_x, self.__y: batch_y}
        return self.sess.run(accuracy, feed_dict) * 100.0


    ''' 使用不同数据 评估模型 '''
    def evaluation(self, data_set, batch_size):
        batch_x, batch_y = data_set.next_batch(batch_size)
        return self.__getAccuracy(batch_x, batch_y)


    def run(self):
        # 生成模型
        self.model()

        # 前向推导，因为使用了 dropout，训练的推导与预测的不一样，得重新推导
        self.inference()

        # 计算 loss
        self.getLoss()

        # 正则化
        self.__loss = self.regularize(self.__loss, self.REGULAR_BETA)

        # 生成训练的 op
        train_op = self.getTrainOp(self.__loss, self.__learningRate, self.globalStep)

        # 初始化所有变量
        self.initVariables()

        print '\nepoch\tloss\t\taccuracy_train\taccuracy_val:'
        ava_loss = 0
        ava_accuracy = 0
        cal_times = 0

        best_accuracy_val = 0
        decrease_acu_val_times = 0

        for step in range(self.__steps):
            batch_x, batch_y = self.__trainSet.next_batch(self.BATCH_SIZE)
            feed_dict = {self.__X: batch_x, self.__y: batch_y}

            self.sess.run(train_op, feed_dict)

            if step % 50 == 0:
                ava_loss += self.sess.run(self.__loss, feed_dict)
                ava_accuracy += self.__getAccuracy(batch_x, batch_y)
                cal_times += 1

            if step % self.__iterPerEpoch == 0 and step != 0:
                epoch = step // self.__iterPerEpoch
                ava_loss /= cal_times
                ava_accuracy /= cal_times

                accuracy_val = self.evaluation(self.__valSet, self.__valSize)

                print '%d\t%.10f\t%.10f%%\t%.6f%%' % (epoch, ava_loss, ava_accuracy, accuracy_val)

                if accuracy_val > best_accuracy_val:
                    best_accuracy_val = accuracy_val
                    decrease_acu_val_times = 0

                    self.saveModel()

                elif accuracy_val <= best_accuracy_val:
                    decrease_acu_val_times += 1
                    if decrease_acu_val_times > 10:
                        break

                cal_times = 0
                ava_loss = 0
                ava_accuracy = 0

        self.restoreModel()

        accuracy_train = self.evaluation(self.__trainSet, self.__trainSize)
        accuracy_val = self.evaluation(self.__valSet, self.__valSize)
        accuracy_test = self.evaluation(self.__testSet, self.__testSize)

        print '\ntraining set accuracy: %.6f%%' % accuracy_train
        print 'validation set accuracy: %.6f%%' % accuracy_val
        print 'test set accuracy: %.6f%%' % accuracy_test


o_nn = BP()
o_nn.run()
