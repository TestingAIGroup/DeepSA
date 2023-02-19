from multiprocessing import Pool
from keras.models import load_model, Model
from utils import *
from utils import load_CIFAR
import os
import keras.backend as K
import numpy as np
import time
from scipy.stats import gaussian_kde
from sadl_variant.innvestigate_lenet5 import get_topk_neurons_composition
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path = 'E:/githubAwesomeCode/1DLTesting/improve_DLtesting/sadl_variant/WISA_paper_code/results/lenet5/important/'

class SurpriseAdequacyDSA:
    # sa = SurpriseAdequacy(model, X_train, layer_names, upper_bound, dataset, topk_neuron_idx)
    def __init__(self,  model, train_inputs, layer_names, upper_bound, dataset, topk_neuron_idx):

        #self.surprise = surprise
        self.model = model
        self.train_inputs = train_inputs
        self.layer_names = layer_names
        self.upper_bound = upper_bound
        self.n_buckets = 1000
        self.dataset = dataset
        self.topk_neuron_idx = topk_neuron_idx
        self.save_path='E:/githubAwesomeCode/1DLTesting/improve_DLtesting/sadl_variant/WISA_paper_code/results/'
        if dataset == 'drive': self.is_classification = False   #处理非分类任务
        else: self.is_classification = True
        self.num_classes = 10
        self.var_threshold = 1e-5

    #sa.test(X_test, approach)
    def test(self, test_inputs, dataset_name,  instance='dsa'):
        if instance == 'dsa':
            print('dataset_name: ', dataset_name)
            target_lsa = fetch_lsa(self.model, self.train_inputs, test_inputs, dataset_name,
                                   self.layer_names, self.num_classes, self.is_classification,
                                   self.save_path, self.dataset, self.topk_neuron_idx, self.var_threshold)
        np.save('E:/githubAwesomeCode/1DLTesting/improve_DLtesting/sadl_variant/WISA_paper_code/results/lenet5/lenet5_lsa_generation_80.npy', target_lsa)

        return target_lsa

def fetch_lsa(model, x_train, x_target, target_name, layer_names, num_classes, is_classification,
              save_path, dataset, topk_neuron_idx, var_threshold):

    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, num_classes,
        is_classification, save_path, dataset, topk_neuron_idx)

    print('train_ats: ', train_ats.shape)
    print('target_ats: ',target_ats.shape)

    class_matrix = {}
    if is_classification:
        for i, label in enumerate(train_pred):
            if label.argmax(axis=-1) not in class_matrix:
                class_matrix[label.argmax(axis=-1)] = []
            class_matrix[label.argmax(axis=-1)].append(i)
        print('yes')
    print(class_matrix.keys())

    kdes, removed_cols = _get_kdes(train_ats, train_pred, class_matrix,
                                   is_classification, num_classes, var_threshold)

    lsa = []
    if is_classification:
        for i, at in enumerate(target_ats):
            label = target_pred[i].argmax(axis=-1)
            kde = kdes[label]
            lsa.append(_get_lsa(kde, at, removed_cols))
    else:
        kde = kdes[0]
        for at in target_ats:
            lsa.append(_get_lsa(kde, at, removed_cols))

    return lsa

def _get_kdes(train_ats, train_pred, class_matrix, is_classification, num_classes, var_threshold):

    is_classification =True
    removed_cols = []
    if is_classification:
        for label in range(num_classes):
            col_vectors = np.transpose(train_ats[class_matrix[label]])
            for i in range(col_vectors.shape[0]):
                if ( np.var(col_vectors[i]) < var_threshold and i not in removed_cols ):
                    removed_cols.append(i)

        kdes = {}
        for label in range(num_classes):
            refined_ats = np.transpose(train_ats[class_matrix[label]])
            refined_ats = np.delete(refined_ats, removed_cols, axis=0)  #(1120, 9665)
            #print('refined_ats: ', refined_ats.shape)

            if refined_ats.shape[0] == 0:
                print("ats were removed by threshold {}".format(var_threshold))
                break
            kdes[label] = gaussian_kde(refined_ats)
    else:
        col_vectors = np.transpose(train_ats)
        for i in range(col_vectors.shape[0]):
            if np.var(col_vectors[i]) < var_threshold:
                removed_cols.append(i)

        refined_ats = np.transpose(train_ats)
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)
        if refined_ats.shape[0] == 0:
            print("ats were removed by threshold {}".format(var_threshold))
        kdes = [gaussian_kde(refined_ats)]

    return kdes, removed_cols

def _get_lsa(kde, at, removed_cols):
    refined_at = np.delete(at, removed_cols, axis=0)
    return np.asscalar(-kde.logpdf(np.transpose(refined_at)))

def _get_train_target_ats(model, x_train, x_target, target_name, layer_names,
                          num_classes, is_classification, save_path, dataset, topk_neuron_idx):

    saved_train_path = _get_saved_path(save_path, dataset, "train", layer_names)

    if os.path.exists(saved_train_path[0]):
        train_ats = np.load(saved_train_path[0]) #train_ats:  (60000, 12)
        train_pred = np.load(saved_train_path[1]) #train_pred:  (60000, 10)

    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            topk_neuron_idx,
            num_classes = num_classes,
            is_classification=is_classification,
            save_path=saved_train_path,
        )

    saved_target_path = _get_saved_path(save_path, dataset, 'cifar10', layer_names)
    #Team DEEPLRP
    if os.path.exists(saved_target_path[0]):
        target_ats = np.load(saved_target_path[0])
        target_pred = np.load(saved_target_path[1])

    else:
        # target就是X_train
        target_ats, target_pred = get_ats(
            model,
            x_target, #X_test
            target_name,
            layer_names,
            topk_neuron_idx,
            num_classes=num_classes,
            is_classification=is_classification,
            save_path=saved_target_path,
        )

    return train_ats, train_pred, target_ats, target_pred

def get_ats( model, dataset, name, layer_names, topk_neuron_idx, save_path=None, batch_size=128, is_classification=True, num_classes=10, num_proc=10,):

    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )

    if is_classification:
        p = Pool(num_proc)  #Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果池还没有满，就会创建一个新的进程来执行请求。
        pred = model.predict(dataset, batch_size=batch_size, verbose=1)

        if len(layer_names) == 1:  #计算coverage的只有一层
            layer_outputs = [temp_model.predict(dataset, batch_size=batch_size, verbose=1)]
        else:
            layer_outputs = temp_model.predict(dataset, batch_size=batch_size, verbose=1)

        ats = None
        for layer_name, layer_output in zip(layer_names, layer_outputs):  # (1, 60000, 4, 4, 12)
            if layer_output[0].ndim == 3:
                list_top_neuron_idx = topk_neuron_idx[layer_name]
                layer_matrix = np.array(p.map(_aggr_output, [layer_output[i][:,:,list_top_neuron_idx] for i in range(len(dataset))]) )

            else:
                list_top_neuron_idx = topk_neuron_idx[layer_name]
                layer_matrix = np.array(layer_output[:, list_top_neuron_idx])

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None
    #
    # if save_path is not None:
    #     np.save(save_path[0], ats)
    #     np.save(save_path[1], pred)

    return ats, pred

def _get_saved_path(base_path, dataset, dtype, layer_names):
    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + dtype + "_pred" + ".npy"),
    )

def get_sc(lower, upper, k, sa):

    buckets = np.digitize(sa, np.linspace(lower, upper, k))
    return len(list(set(buckets))) / float(k) * 100

def find_closest_at(at, train_ats):
    #The closest distance between subject AT and training ATs.

    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])

def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]


# def get_topk_neurons_composition(model, num_pro):
#     all_total_R_act2 = np.load(path + 'lenet4_act2.npy')
#     all_total_R_act3 = np.load(path + 'lenet4_act3.npy')
#     all_total_R_act4 = np.load(path + 'lenet4_act4.npy')
#     all_total_R_act5 = np.load(path + 'lenet4_act5.npy')
#     all_total_R_act7 = np.load(path + 'lenet4_act7.npy')
#     # all_total_R_act8 = np.load(path + 'lenet4_act8.npy')
#
#     total_topk_neuron_idx = {}
#
#     neuron_outs_act2 = np.zeros((all_total_R_act2.shape[-1],))
#     for i in range(all_total_R_act2.shape[-1]):
#         neuron_outs_act2[i] = np.mean(all_total_R_act2[..., i])
#     num_relevant_neurons_act2 = round(num_pro * len(neuron_outs_act2))
#
#     neuron_outs_act3 = np.zeros((all_total_R_act3.shape[-1],))
#     for i in range(all_total_R_act3.shape[-1]):
#         neuron_outs_act3[i] = np.mean(all_total_R_act3[..., i])
#     num_relevant_neurons_act3 = round(num_pro * len(neuron_outs_act3))
#
#     neuron_outs_act4 = np.zeros((all_total_R_act4.shape[-1],))
#     for i in range(all_total_R_act4.shape[-1]):
#         neuron_outs_act4[i] = np.mean(all_total_R_act4[..., i])
#     num_relevant_neurons_act4 = round(num_pro * len(neuron_outs_act4))
#
#     neuron_outs_act5 = np.zeros((all_total_R_act5.shape[-1],))
#     for i in range(all_total_R_act5.shape[-1]):
#         neuron_outs_act5[i] = np.mean(all_total_R_act5[..., i])
#     num_relevant_neurons_act5 = round(num_pro * len(neuron_outs_act5))
#
#     neuron_outs_act7 = np.zeros((all_total_R_act7.shape[-1],))
#     for i in range(all_total_R_act7.shape[-1]):
#         neuron_outs_act7[i] = np.mean(all_total_R_act7[..., i])
#     num_relevant_neurons_act7 = round(num_pro * len(neuron_outs_act7))
#
#     # neuron_outs_act8 = np.zeros((all_total_R_act8.shape[-1],))
#     # for i in range(all_total_R_act8.shape[-1]):
#     #     neuron_outs_act8[i] = np.mean(all_total_R_act8[..., i])
#     # num_relevant_neurons_act8 = round(num_pro * len(neuron_outs_act8))
#
#     # 在python3.6里是activation5对应的是model.layers[12].name
#     # 在python3.6里是activation6对应的是model.layers[14].name
#     # 但是使用python3.8跑的sadl_improve文件，所以应该用11和13
#
#     #  根据proportion取top k important neurons
#     top_k_neuron_indexes_act2 = (
#     np.argsort(neuron_outs_act2, axis=None)[-num_relevant_neurons_act2:len(neuron_outs_act2)])
#     total_topk_neuron_idx[model.layers[1].name] = top_k_neuron_indexes_act2
#
#     top_k_neuron_indexes_act3 = (
#     np.argsort(neuron_outs_act3, axis=None)[-num_relevant_neurons_act3:len(neuron_outs_act3)])
#     total_topk_neuron_idx[model.layers[2].name] = top_k_neuron_indexes_act3
#
#     top_k_neuron_indexes_act4 = (
#         np.argsort(neuron_outs_act4, axis=None)[-num_relevant_neurons_act4:len(neuron_outs_act4)])
#     total_topk_neuron_idx[model.layers[3].name] = top_k_neuron_indexes_act4
#
#     top_k_neuron_indexes_act5 = (
#         np.argsort(neuron_outs_act5, axis=None)[-num_relevant_neurons_act5:len(neuron_outs_act5)])
#     total_topk_neuron_idx[model.layers[4].name] = top_k_neuron_indexes_act5
#
#     top_k_neuron_indexes_act7 = (
#         np.argsort(neuron_outs_act7, axis=None)[-num_relevant_neurons_act7:len(neuron_outs_act7)])
#     total_topk_neuron_idx[model.layers[6].name] = top_k_neuron_indexes_act7
#
#     # top_k_neuron_indexes_act8 = (
#     #     np.argsort(neuron_outs_act8, axis=None)[-num_relevant_neurons_act8:len(neuron_outs_act8)])
#     # total_topk_neuron_idx[model.layers[7].name] = top_k_neuron_indexes_act8
#
#     print('total_topk_neuron_idx: ', total_topk_neuron_idx)
#     return total_topk_neuron_idx

def preprocess_image(img_path, target_size=(28, 28)):
    img2 = Image.open(img_path) # 读取图片
    input_img_data = np.array(img2) # 变成nparray

    return input_img_data

if __name__ == "__main__":
    #
    # load the data:
    dataset = 'mnist'
    X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)  # 在utils中修改
    print('X_test: ', X_test.shape)

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

    upper_bound = 2000

    num_pros = 0.8
    total_topk_neuron_idx = get_topk_neurons_composition(model, num_pros)
    print('total_topk_neuron_idx: ', total_topk_neuron_idx)

    layer_names = []
    for key in total_topk_neuron_idx.keys():
        layer_names.append(key)
    print('layer_names: ', layer_names)

    inputs_file_folder = 'E:/githubAwesomeCode/1DLTesting/improve_DLtesting/sadl_variant/WISA_paper_code/results/generated_inputs_mnist_lenet1_lenet5/DeepXplore/'

    imgs = os.listdir(inputs_file_folder)
    new_test = []
    for img in imgs:
        new_test.append(preprocess_image(inputs_file_folder + img))
    new_test = np.array(new_test)

    img_shape = 28, 28, 1
    adv_generation = new_test.reshape(new_test.shape[0], *img_shape).astype('float32') / 255

    sa = SurpriseAdequacyDSA(model, X_train, layer_names, upper_bound, dataset, total_topk_neuron_idx)
    sa.test(adv_generation, dataset)

