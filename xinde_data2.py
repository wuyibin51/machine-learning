from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import obspy
import pandas
from obspy import read
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPool1D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import datetime
import time

starttime = datetime.datetime.now()

#long running 计时
time.sleep(2)

np.set_printoptions(threshold=np.inf)

train_path = './mnist_image_label/mydata/'
train_txt = './mnist_image_label/myfile.txt'
x_train_savepath = './mnist_image_label/x_data.npy'
y_train_savepath = './mnist_image_label/y_data.npy'

x_test_savepath = './mnist_image_label/x_pre.npy'
y_test_savepath = './mnist_image_label/y_pre.npy'
x_test = np.load(x_test_savepath)
y_test = np.load(y_test_savepath)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1))

def generateds(path, txt):
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        wave_path = path + value[0]  # 拼出波形路径和文件名
        st = read(wave_path)  # 读入接收函数
        tr = st[0]

        tr = np.array(tr)  # 接收函数变为np.array格式
        if ~np.isnan(tr).any(axis=0):    # 将含空值的文件删除
            x.append(tr)  # 归一化后的数据，贴到列表x
            y_.append(value[1])  # 标签贴到列表y_
            print('loading : ' + content)  # 打印状态提示


    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath):
    print('-------------Load Datasets-----------------')
    x_train = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_path, train_txt)
    print('-------------Save Datasets-----------------')

    np.save(x_train_savepath, x_train)
    np.save(y_train_savepath, y_train)


np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

# reshape input to be 3D [samples, timesteps, features]
#x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],1))
print(x_train.shape, y_train.shape)

#全连接神经网络
class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()



        self.flatten = Flatten()
        self.g1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.p2 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.p3 = Dense(60, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.p4 = Dense(30, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())

        self.f3 = Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):

        x = self.flatten(x)
        x = self.g1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)

        y = self.f3(x)
        return y

#卷积神经网络
# class LeNet5(Model):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#
#         self.d1 = Conv1D(16, 5, activation='relu')
#         self.e1 = MaxPool1D(2)
#         self.e11 = Conv1D(16, 5, activation='relu')
#         self.p1 = MaxPool1D(2)
#
#
#         self.flatten = Flatten()
#         self.g1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
#         self.p2 = Dropout(0.5)
#         self.p3 = Dense(60, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
#         self.p4 = Dropout(0.5)
#
#         self.f3 = Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
#
#     def call(self, x):
#
#         x = self.d1(x)
#         x = self.e1(x)
#         x = self.e11(x)
#         x = self.p1(x)
#         x = self.flatten(x)
#         x = self.g1(x)
#         x = self.p2(x)
#         x = self.p3(x)
#         x = self.p4(x)
#
#         y = self.f3(x)
#         return y

#循环神经网络RNN
# class MyNet(Model):
#     def __init__(self):
#         super(MyNet, self).__init__()
#
#         self.d1 = LSTM(80, return_sequences=True)
#         self.e1 = Dropout(0.2)
#         self.e11 = LSTM(100)
#         self.p1 = Dropout(0.2)
#         self.g1 = Dense(2)

#     def call(self, x):
#
#         x = self.d1(x)
#         x = self.e1(x)
#         x = self.e11(x)
#         x = self.p1(x)
#         x = self.flatten(x)
#         x = self.g1(x)
#         return y

model = LeNet5()

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/MyLeNet5.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(x_train, y_train, batch_size=64, epochs=30, validation_split=0.3, validation_freq=1, callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
file = open('./weights0.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################
endtime = datetime.datetime.now()
str="run time: %d seconds" % ((endtime - starttime).seconds)
print(str)
# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

result = model.predict(x_test)
y_predict = tf.argmax(result,axis=1)
print("测试集标签y_test")
print(y_test)
print("预测结果y_predict")
print(y_predict)

report = classification_report(y_test,y_predict,labels=[0,1],target_names=["hao","huai"])
print("分类结果—混淆矩阵")
print(report)
report2 = roc_auc_score(y_test,y_predict)
print("AUC值")
print(report2)