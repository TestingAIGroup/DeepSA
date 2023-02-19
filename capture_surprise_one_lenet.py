import numpy as np
from utils import load_CIFAR
from keras.models import load_model, Model
import os
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":
    # load the data:
    dataset = 'mnist'
    X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)  # 在utils中修改
    img_rows, img_cols = 28, 28

    # set the model
    model_path = r'E:/githubAwesomeCode/1DLTesting/sadl_improve/neural_networks/lenet5'
    model_name = (model_path).split('/')[-1]
    print(model_name)

    try:
        json_file = open(model_path + '.json', 'r')  # Read Keras model parameters (stored in JSON file)
        file_content = json_file.read()
        json_file.close()

        model = model_from_json(file_content)
        model.load_weights(model_path + '.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    except:
        model = load_model(model_path + '.h5')
    model.summary()

    print('X_test: ', X_test.shape) #(10000, 28, 28, 1)
    print('Y_test: ', Y_test.shape) #(10000, 10)


    # lid_values = np.load('E:/githubAwesomeCode/1DLTesting/improve_DLtesting/sadl_variant/WISA_paper_code/results/lenet5/lenet5_dsa_ats_20.npy')
   #  lid_values = np.load('E:/original data/LID/lenet5/lenet5_80_5000_200.npy')

    lid_values = np.load('E:/githubAwesomeCode/1DLTesting/improve_DLtesting/sadl_variant/WISA_paper_code/results/lenet5/lenet5_lsa_ats_90.npy')
    ascending_index = np.argsort(lid_values)  # 从小到大
    descending_index = np.argsort(lid_values)[::-1]  # 从大到小
    print('ascending_index: ', ascending_index)
    print('descending_index: ', descending_index)

    img_num=[100, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
             5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]

    descending_accuracy= {}; ascending_accuracy={}; random_accuracy={}
    for num in img_num:
        # 计算descending LID values 的accuracy
        des_evaluate_data = X_test[descending_index[0:num], :, :, :]
        des_evaluate_label = Y_test[descending_index[0:num], :]
        loss, des_accuracy = model.evaluate(des_evaluate_data, des_evaluate_label)
        descending_accuracy[num]=des_accuracy
        print('descending_accuracy: ', des_accuracy)

        # # 计算ascending LID values 的accuracy
        # asc_evaluate_data = X_test[ascending_index[0:num], :, :, :]
        # asc_evaluate_label = Y_test[ascending_index[0:num], :]
        # loss, asc_accuracy = model.evaluate(asc_evaluate_data, asc_evaluate_label)
        # ascending_accuracy[num] = asc_accuracy
        # print('ascending_accuracy: ', ascending_accuracy)
        #
        # # 计算random LID values 的accuracy
        # np.random.shuffle(ascending_index)  #打乱ascending_index,相当于随机选择
        # ran_evaluate_data = X_test[ascending_index[0:num], :, :, :]
        # ran_evaluate_label = Y_test[ascending_index[0:num], :]
        # loss, ran_accuracy = model.evaluate(ran_evaluate_data, ran_evaluate_label)
        # random_accuracy[num] = ran_accuracy
        # print('accuracy', ran_accuracy)

    print('descending_accuracy: ', descending_accuracy)
    # print('ascending_accuracy: ', ascending_accuracy)
    # print('random_accuracy: ', random_accuracy)