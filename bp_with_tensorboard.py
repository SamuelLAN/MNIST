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
 稍微复杂点的 BP
 具有四层网络，一层输入，两层隐藏，一层输出
 使用了 TensorBoard 可视化
 小技巧：
    学习率自动下降
    dropout
    regularize
 准确率：
    training set accuracy: 0.981909%
    validation set accuracy: 0.977000%
    test set accuracy: 0.973800%
'''
class BP(base.NN):
    MODEL_NAME = 'bp_deep'                          # 模型的名称

    BATCH_SIZE = 128                                # 迭代的 epoch 次数
    EPOCH_TIMES = 200                               # 随机梯度下降的 batch 大小

    IMAGE_SIZE = 28                                 # 输入图片的大小
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
    NUM_CLASSES = 10                                # 输出的类别
    SHAPE_LIST = [(IMAGE_PIXELS, 1024), (1024, 784), (784, 784), (784, NUM_CLASSES)]

    BASE_LEARNING_RATE = 0.01                       # 初始 学习率
    DECAY_RATE = 0.97                               # 学习率 的 下降速率

    DROPOUT_LIST = [0.6, 0.6, 0.6]                  # dropout 的比例
    REGULAR_BETA = 0.01                             # 正则化的 beta 参数


    ''' 自定义 初始化变量 过程 '''
    def init(self):
        # 加载数据
        self.load()

        # 常量
        self.__iterPerEpoch = int(self.__trainSize // self.BATCH_SIZE)
        self.__steps = self.EPOCH_TIMES * self.__iterPerEpoch

        # 输入 与 label
        self.__X = tf.placeholder('float', [None, self.SHAPE_LIST[0][0]], name='X')
        self.__y = tf.placeholder('float', [None, self.SHAPE_LIST[-1][-1]], name='y')
        self.__size = tf.placeholder('float', name='size')
        # 用于预测
        self.__preX = tf.placeholder('float', [None, self.SHAPE_LIST[0][0]], name='preX')
        self.__preY = tf.placeholder('float', [None, self.SHAPE_LIST[-1][-1]], name='preY')
        self.__preSize = tf.placeholder('float', name='preSize')

        self.__accuracyVal = tf.placeholder('float', name='accuracy_val')
        tf.summary.scalar('validation accuracy', self.__accuracyVal)

        # 随训练次数增多而衰减的学习率
        self.__learningRate = self.getLearningRate(
            self.BASE_LEARNING_RATE, self.globalStep, self.BATCH_SIZE, self.__steps, self.DECAY_RATE
        )


    ''' 加载数据 '''
    def load(self):
        self.echo('\nLoading data ... ')

        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.__trainSet = mnist.train
        self.__valSet = mnist.validation
        self.__testSet = mnist.test

        self.__trainSize = self.__trainSet.images.shape[0]
        self.__valSize = self.__valSet.images.shape[0]
        self.__testSize = self.__testSet.images.shape[0]

        self.echo('Finish loading ')


    ''' 模型 '''
    def model(self):
        self.__output = self.fullConnectModel(self.__X, self.SHAPE_LIST)


    ''' 前向推导 '''
    def inference(self):
        self.__predict = self.fullConnectModel(self.__preX, self.SHAPE_LIST, False)


    ''' 计算 loss '''
    def getLoss(self):
        with tf.name_scope('loss'):
            self.__loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.__output, labels=self.__y)
            )


    ''' 计算准确率 '''
    @staticmethod
    def __getAccuracy(labels, predict, size, name = ''):
        with tf.name_scope('accuracy'):
            labels = tf.argmax(labels, 1)
            predict = tf.argmax(predict, 1)
            correct = tf.equal(labels, predict)                                     # 返回 predict 与 labels 相匹配的结果

            accuracy = tf.divide(tf.reduce_sum(tf.cast(correct, tf.float32)), size) # 计算准确率

            if name:
                tf.summary.scalar('%s accuracy' % name, accuracy)                   # 将 准确率 记录到 TensorBoard

            return accuracy


    ''' 使用不同数据 评估模型 '''
    def evaluation(self
                   , data_set, batch_size, name = ''):
        with tf.name_scope('evaluation'):
            batch_x, batch_y = data_set.next_batch(batch_size)
            accuracy = self.__getAccuracy(self.__preY, self.__predict, self.__preSize, name)
            return self.sess.run(accuracy, {self.__preX: batch_x, self.__preY: batch_y, self.__preSize: batch_x.shape[0]})


    def run(self):
        # 生成模型
        self.model()

        # 前向推导，因为使用了 dropout，训练的推导与预测的不一样，得重新推导
        self.inference()

        # 计算 loss
        self.getLoss()

        # 正则化
        # self.__loss = self.regularize(self.__loss, self.REGULAR_BETA)

        # 生成训练的 op
        train_op = self.getTrainOp(self.__loss, self.__learningRate, self.globalStep)

        # 用于 TensorBoard 查看准确率
        accuracy_train = self.__getAccuracy(self.__y, self.__output, self.__size, name='training')
        # accuracy_val = self.__getAccuracy(self.__preY, self.__predict, self.__preSize, name='validation')

        # 初始化所有变量
        self.initVariables()

        # TensorBoard merge summary
        self.mergeSummary()

        best_accuracy_val = 0       # 校验集准确率 最好的情况
        decrease_acu_val_times = 0  # 校验集准确率连续下降次数

        print '\nepoch:'

        for step in range(self.__steps):
            batch_x, batch_y = self.__trainSet.next_batch(self.BATCH_SIZE)
            feed_dict = {self.__X: batch_x, self.__y: batch_y}

            self.sess.run(train_op, feed_dict)

            if step % self.__iterPerEpoch == 0 and step != 0:
                epoch = step // self.__iterPerEpoch
                accuracy_val = self.evaluation(self.__valSet, self.__valSize) # 获取校验集准确率
                print '%d\t%.10f%%' % (epoch, accuracy_val) # 输出进度

                # 将数据记录到 TensorBoard
                feed_dict = {self.__X: batch_x, self.__y: batch_y, self.__size: batch_x.shape[0],
                             self.__accuracyVal: accuracy_val}
                # batch_val_x, batch_val_y = self.__valSet.next_batch(self.__valSize)
                # feed_dict = {self.__X: batch_x, self.__y: batch_y, self.__size: batch_x.shape[0],
                #              self.__preX: batch_val_x, self.__preY: batch_val_y, self.__preSize: batch_val_x.shape[0]}
                self.addSummary(feed_dict, epoch)

                if accuracy_val > best_accuracy_val:    # 若校验集准确率 比 最高准确率高
                    best_accuracy_val = accuracy_val
                    decrease_acu_val_times = 0

                    self.saveModel()                    # 保存模型

                else:                                   # 否则
                    decrease_acu_val_times += 1
                    if decrease_acu_val_times > 10:
                        break

        self.closeSummary()  # 关闭 TensorBoard

        self.restoreModel()  # 恢复模型

        accuracy_train = self.evaluation(self.__trainSet, self.__trainSize)
        accuracy_val = self.evaluation(self.__valSet, self.__valSize)
        accuracy_test = self.evaluation(self.__testSet, self.__testSize)

        print '\ntraining set accuracy: %.6f%%' % accuracy_train
        print 'validation set accuracy: %.6f%%' % accuracy_val
        print 'test set accuracy: %.6f%%' % accuracy_test


o_nn = BP()
o_nn.run()
