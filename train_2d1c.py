#!/usr/bin/env python3

import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import callbacks

if __name__ == "__main__":

    # read train dataset
    # 读训练数据集
    train_dataset = pd.read_hdf("data/ujiindoorloc/saved/training_df.h5")
    train_waps_rssis = np.array(train_dataset["WAPs_RSSIs"])
    train_building = np.array(train_dataset["BUILDINGID"])
    train_floor = np.array(train_dataset["FLOOR"])
    train_space = np.array(train_dataset["SPACEID"])

    # create an array full of no signal WI-FI for later use
    # 创建一个全是无信号的train_x
    train_x = np.array([[[0, -110] for i in range(51)] for j in range(len(train_waps_rssis))])

    # copy data
    # 复制数据
    for i in range(len(train_waps_rssis)):
        for j in range(len(train_waps_rssis[i])):
            if j > (len(train_x[i]) - 1):
                break
            train_x[i][j][0] = train_waps_rssis[i][j][0] + 1  # 这里+1是因为无信号是0, 剩下的向后顺延
            train_x[i][j][1] = train_waps_rssis[i][j][1]

    # convert train_y to one-hot encoding
    # 转成one-hot编码
    # 创建全是0的train_y
    train_y = np.zeros((len(train_building), 8))
    for i in range(len(train_building)):
        train_y[i][int(train_building[i])] = 1
        train_y[i][int(train_floor[i]) + 3] = 1

    # to fp32
    # 转 float
    train_x = train_x.astype('float32')
    train_y = train_y.astype('float32')

    # normalize
    train_x[::, ::, 0] /= 521
    train_x[::, ::, 1] += 110
    train_x[::, ::, 1] /= 110

    # reshape from [none, 52, 2] to [none, 52, 2, 1]
    # 把[none, 52, 2] 变成 [none, 52, 2, 1]

    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))


    # # read val dataset
    # # 读测试数据集
    val_dataset = pd.read_hdf("data/ujiindoorloc/saved/validation_df.h5")
    val_waps_rssis = np.array(val_dataset["WAPs_RSSIs"])
    val_building = np.array(val_dataset["BUILDINGID"])
    val_floor = np.array(val_dataset["FLOOR"])

    # create an array full of no signal WI-FI for later use
    # 创建一个全是无信号的val_x
    val_x = np.array([[[0, -110] for i in range(51)] for j in range(len(val_waps_rssis))])

    # copy data
    # 复制数据
    for i in range(len(val_waps_rssis)):
        for j in range(len(val_waps_rssis[i])):
            if j > (len(val_x[i]) - 1):
                break
            val_x[i][j][0] = val_waps_rssis[i][j][0] + 1  # 这里+1是因为无信号是0, 剩下的向后顺延
            val_x[i][j][1] = val_waps_rssis[i][j][1]

    # convert val_y to one-hot encoding
    # 转成one-hot编码
    # 创建全是0的val_y
    val_y = np.zeros((len(val_building), 8))
    for i in range(len(val_building)):
        val_y[i][int(val_building[i])] = 1
        val_y[i][int(val_floor[i]) + 3] = 1

    # to fp32
    # 转 float
    val_x = val_x.astype('float32')
    val_y = val_y.astype('float32')

    # normalize
    val_x[::, ::, 0] /= 521
    val_x[::, ::, 1] += 110
    val_x[::, ::, 1] /= 110

    # reshape from [none, 52, 2] to [none, 52, 2, 1]
    # 把[none, 52, 2] 变成 [none, 52, 2, 1]

    val_x = val_x.reshape((val_x.shape[0], val_x.shape[1], val_x.shape[2], 1))


    # new model
    # 创建模型
    model = keras.Sequential()

    # add conv
    # 卷积层
    model.add(layers.Conv2D(16, kernel_size=(3, 1), activation='relu',
                            input_shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3])))

    model.add(layers.Conv2D(16, kernel_size=(3, 2), activation='relu'))

    # dropout
    # 随机断开连接
    model.add(layers.Dropout(0.25, name="droupout-0"))

    # flat
    # 展平
    model.add(layers.Flatten())

    # classifier
    # 分类器
    classifier_hidden_layers = [64, 128]
    for i in range(len(classifier_hidden_layers)):
        model.add(layers.Dense(classifier_hidden_layers[i], name="classifier-hidden-" + str(i), activation='relu'))
        model.add(layers.Dropout(0.25, name="droupout-" + str(i + 1)))

    model.add(layers.Dense(8, name="activation-0", activation='sigmoid'))  # 'sigmoid' for multi-label classification

    # summary
    # 展示模型
    model.summary()

    # compile
    # 编译模型
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    # train the model
    # 训练模型
    callback = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
    model.fit(train_x, train_y, batch_size=4, epochs=100, verbose=1, validation_data=(val_x, val_y),
              callbacks=[callback])
    # save the model for later use
    # 保存模型
    model.save("model_2d1c")

    # 使用网络预测训练集
    predict_train = model.predict(train_x)

    # 正确率变量
    acc_building = 0
    acc_floor = 0

    # 循环判断正确
    for i in range(len(train_y)):
        if predict_train[i, :3].argmax() == train_y[i, :3].argmax():
            acc_building += 1
        if predict_train[i, 3:].argmax() == train_y[i, 3:].argmax():
            acc_floor += 1

    print("train building hit ratio: %.2f" % (acc_building / len(train_y)))
    print("train floor hit ratio %.2f" % (acc_floor / len(train_y)))

    # 使用网络预测测试集
    predict_val = model.predict(val_x)

    # 正确率变量
    acc_building = 0
    acc_floor = 0

    # 循环判断正确
    for i in range(len(val_y)):
        if predict_val[i, :3].argmax() == val_y[i, :3].argmax():
            acc_building += 1
        if predict_val[i, 3:].argmax() == val_y[i, 3:].argmax():
            acc_floor += 1

    print("val building hit ratio: %.2f" % (acc_building / len(val_y)))
    print("val floor hit ratio %.2f" % (acc_floor / len(val_y)))
