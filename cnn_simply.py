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
 简单的卷积神经网络
 网络结构：
    输入
    卷积
    pooling
    卷积
    pooling
    全连接
    全连接
    输出
 准确率：
    99.3%
'''
class CNN(base.NN):
    MODEL_NAME = 'cnn_simply'                       # 模型的名称

    BATCH_SIZE = 128                                # 迭代的 epoch 次数
    EPOCH_TIMES = 200                               # 随机梯度下降的 batch 大小

    IMAGE_SIZE = 28                                 # 输入图片的大小
    IMAGE_SHAPE = [IMAGE_SIZE, IMAGE_SIZE]
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
    NUM_CHANNEL = 1                                 # 输入图片为 1 通道，灰度值
    NUM_CLASSES = 10                                # 输出的类别

    BASE_LEARNING_RATE = 0.01                       # 初始 学习率
    DECAY_RATE = 0.9                                # 学习率 的 下降速率

    REGULAR_BETA = 0.01                             # 正则化的 beta 参数
    MOMENTUM = 0.9                                  # 动量的大小

    TENSORBOARD_SHOW_IMAGE = True                  # 默认不将 image 显示到 TensorBoard，以免影响性能

    MODEL = [                                       # 深度模型
        {
            'type': 'conv',
            'shape': [NUM_CHANNEL, 32],
            'k_size': [5, 5],
        },
        {
            'type': 'pool',
            'k_size': [2, 2],
        },
        {
            'type': 'conv',
            'shape': [32, 64],
            'k_size': [2, 2],
        },
        {
            'type': 'pool',
            'k_size': [2, 2],
        },
        {
            'type': 'fc',
            'shape': [ (IMAGE_SIZE / 2 / 2) ** 2 * 64, 1024 ],
        },
        {
            'type': 'dropout',
            'keep_prob': 0.5,
        },
        {
            'type': 'fc',
            'shape': [1024, NUM_CLASSES]
        }
    ]


    ''' 自定义 初始化变量 过程 '''
    def init(self):
        # 加载数据
        self.load()

        # 常量
        self.__iterPerEpoch = int(self.__trainSize // self.BATCH_SIZE)
        self.__steps = self.EPOCH_TIMES * self.__iterPerEpoch

        # 输入 与 label
        self.__X = tf.placeholder(tf.float32, [None, self.IMAGE_PIXELS], name='X')
        self.__y = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='y')
        self.__size = tf.placeholder(tf.float32, name='size')

        # 用于预测
        self.__preX = tf.placeholder(tf.float32, [None, self.IMAGE_PIXELS], name='preX')
        self.__preY = tf.placeholder(tf.float32, [None, self.NUM_CLASSES], name='preY')
        self.__preSize = tf.placeholder(tf.float32, name='preSize')

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
        self.__output = self.deepModel(self.__X, self.IMAGE_SHAPE)


    ''' 前向推导 '''
    def inference(self):
        with tf.name_scope('inference'):
            self.__predict = self.deepModel(self.__preX, self.IMAGE_SHAPE, False)


    ''' 计算 loss '''
    def getLoss(self):
        with tf.name_scope('loss'):
            self.__loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.__output, labels=self.__y)
            )


    ''' 获取 train_op '''
    def getTrainOp(self, loss, learning_rate, global_step):
        tf.summary.scalar('loss', loss)     # TensorBoard 记录 loss

        with tf.name_scope('optimizer'):
            optimizer = tf.train.MomentumOptimizer(learning_rate, self.MOMENTUM)
            return optimizer.minimize(loss, global_step=global_step)


    ''' 计算准确率 '''
    @staticmethod
    def __getAccuracy(labels, predict, _size, name = ''):
        if name:
            scope_name = '%s_accuracy' % name
        else:
            scope_name = 'accuracy'

        with tf.name_scope(scope_name):
            labels = tf.argmax(labels, 1)
            predict = tf.argmax(tf.nn.softmax(predict), 1)  # softmax 是单调增函数，当使用 argmax 时可以不做 softmax 操作
            correct = tf.equal(labels, predict) # 返回 predict 与 labels 相匹配的结果

            accuracy = tf.divide( tf.reduce_sum( tf.cast(correct, tf.float32) ), _size ) # 计算准确率
            if name: # 将 准确率 记录到 TensorBoard
                tf.summary.scalar('accuracy', accuracy)

            return accuracy


    ''' 使用不同数据 评估模型 '''
    def evaluation(self, data_set, batch_size, accuracy = None):
        batch_x, batch_y = data_set.next_batch(batch_size)
        return self.sess.run(accuracy, {
            self.__preX: batch_x, self.__preY: batch_y, self.__preSize: batch_x.shape[0]
        })


    def run(self):
        # 生成模型
        self.model()

        # 前向推导，用于预测准确率 以及 TensorBoard 里能同时看到 训练集、校验集 的准确率
        self.inference()

        # 计算 loss
        self.getLoss()

        # 正则化
        # self.__loss = self.regularize(self.__loss, self.REGULAR_BETA)

        # 生成训练的 op
        train_op = self.getTrainOp(self.__loss, self.__learningRate, self.globalStep)

        ret_accuracy_train = self.__getAccuracy(self.__y, self.__output, self.__size, name='training')
        ret_accuracy_val = self.__getAccuracy(self.__preY, self.__predict, self.__preSize, name='validation')

        # 初始化所有变量
        self.initVariables()

        # TensorBoard merge summary
        self.mergeSummary()

        best_accuracy_val = 0           # 校验集准确率 最好的情况
        decrease_acu_val_times = 0      # 校验集准确率连续下降次数

        # 获取校验集的数据，用于之后获取校验集准确率，不需每次循环重新获取
        batch_val_x, batch_val_y = self.__valSet.next_batch(self.__valSize)
        batch_val_size = batch_val_x.shape[0]

        print '\nepoch\taccuracy_val:'

        for step in range(self.__steps):
            if step % 50 == 0:                          # 输出进度
                epoch_progress = 1.0 * step % self.__iterPerEpoch / self.__iterPerEpoch * 100.0
                step_progress = 1.0 * step / self.__steps * 100.0
                self.echo('step: %d (%d|%.2f%%) / %d|%.2f%%     \r' % (step, self.__iterPerEpoch, epoch_progress,
                                                                       self.__steps, step_progress), False)

            batch_x, batch_y = self.__trainSet.next_batch(self.BATCH_SIZE)
            self.sess.run(train_op, {self.__X: batch_x, self.__y: batch_y})     # 运行 训练

            if step % self.__iterPerEpoch == 0 and step != 0: # 完成一个 epoch 时
                epoch = step // self.__iterPerEpoch     # 获取这次 epoch 的 index
                accuracy_val = self.evaluation(self.__valSet, self.__valSize, ret_accuracy_val) # 获取校验集准确率
                print '\n%d\t%.10f%%' % (epoch, accuracy_val)

                feed_dict = {self.__X   : batch_x,      self.__y    : batch_y,      self.__size     : batch_x.shape[0],
                             self.__preX: batch_val_x,  self.__preY : batch_val_y,  self.__preSize  : batch_val_size}
                self.addSummary(feed_dict, epoch)       # 输出数据到 TensorBoard

                if accuracy_val > best_accuracy_val:    # 若校验集准确率 比 最高准确率高
                    best_accuracy_val = accuracy_val
                    decrease_acu_val_times = 0

                    self.saveModel()                    # 保存模型

                else:                                   # 否则
                    decrease_acu_val_times += 1
                    if decrease_acu_val_times > 10:
                        break

        self.closeSummary() # 关闭 TensorBoard

        self.restoreModel() # 恢复模型

        # 计算 训练集、校验集、测试集 的准确率
        accuracy_train = self.evaluation(self.__trainSet, self.__trainSize, ret_accuracy_val)
        accuracy_val = self.evaluation(self.__valSet, self.__valSize, ret_accuracy_val)
        accuracy_test = self.evaluation(self.__testSet, self.__testSize, ret_accuracy_val)

        print '\ntraining set accuracy: %.6f%%' % accuracy_train
        print 'validation set accuracy: %.6f%%' % accuracy_val
        print 'test set accuracy: %.6f%%' % accuracy_test


o_nn = CNN()
o_nn.run()

