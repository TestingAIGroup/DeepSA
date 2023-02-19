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
            target_dsa = fetch_dsa(self.model, self.train_inputs, test_inputs, dataset_name,
                                   self.layer_names, self.num_classes, self.is_classification,
                                   self.save_path, self.dataset, self.topk_neuron_idx)
            np.save('E:/githubAwesomeCode/1DLTesting/improve_DLtesting/sadl_variant/WISA_paper_code/results/lenet5/lenet5_dsa_generation_80.npy', target_dsa)
        # return target_dsa

def fetch_dsa(model, x_train, x_target, target_name, layer_names, num_classes, is_classification, save_path, dataset, topk_neuron_idx):

    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, num_classes,
        is_classification, save_path, dataset, topk_neuron_idx)

    print('train_ats: ', train_ats.shape)
    print('target_ats: ',target_ats.shape)
    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label.argmax(axis=-1) not in class_matrix:
            class_matrix[label.argmax(axis=-1)] = []
        class_matrix[label.argmax(axis=-1)].append(i)
        all_idx.append(i)

    dsa = []

    for i, at in enumerate(target_ats):
        label = target_pred[i].argmax(axis=-1)
        #print('train_ats[class_matrix[label]]: ', train_ats[class_matrix[label]].shape, train_ats[class_matrix[label]])
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
        b_dist, _ = find_closest_at(a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))])
        dsa.append(a_dist / b_dist)

    return dsa


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


def preprocess_image(img_path, target_size=(28, 28)):
    img2 = Image.open(img_path) # 读取图片
    input_img_data = np.array(img2) # 变成nparray

    return input_img_data

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

    _, accuracy = model.evaluate(X_test, Y_test)
    print('original accuracy: ', accuracy)

    upper_bound = 2000

    num_pros = 0.8
    total_topk_neuron_idx = get_topk_neurons_composition(model, num_pros)
    print('total_topk_neuron_idx: ', total_topk_neuron_idx)

    upper_bound = 2
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


