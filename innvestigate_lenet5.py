import innvestigate.utils
from utils import *
from utils import  filter_correct_classifications

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
path = 'E:/githubAwesomeCode/1DLTesting/improve_DLtesting/sadl_variant/WISA_paper_code/results/lenet5/important/'


def get_total_relevance(model, X_train, Y_train):
    analyzer = innvestigate.create_analyzer("lrp.epsilon", model, reverse_keep_tensors=True)
    #analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRP(model, reverse_keep_tensors=True)
    X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model, X_train, Y_train)
    print("X_train_corr shape: ", X_train_corr.shape)  # (88880, 28, 28, 1)

    # activation_8 [21] and activation_6
    analysis = analyzer.analyze(X_train_corr[0:1])
    print('X_train_corr[0:1].shape: ', X_train_corr[0:1].shape)
    for k, v in analyzer._reversed_tensors:
        print("k: ", k)
        print("v.shape",v.shape)
        print('-------------------------')

    all_total_R_dense = np.zeros(analyzer._reversed_tensors[8][1].shape)
    #max_pool = np.zeros(analyzer._reversed_tensors[16][1].shape)    #max_pool:  (1, 4, 4, 128)
    print('all_total_R: ', all_total_R_dense.shape)
   # print('max_pool: ', max_pool.shape)

    for inp in range(len(X_train_corr)):

        analysis = analyzer.analyze(X_train_corr[inp*1:inp*1+1])
        relevance_dense = analyzer._reversed_tensors[8][1]

        all_total_R_dense += relevance_dense

    print('all_total_R_dense: ', all_total_R_dense)

    np.save('sadl_lid/mnist_all_total_R_peultimate_layer.npy', all_total_R_dense)


def get_topk_neurons(model, num_pro):
    all_total_R_dense = np.load('sadl_lid/all_total_R_act8_33.npy')

    print('all_total_R_dense: ', all_total_R_dense)

    # activation8
    neuron_outs_dense = np.zeros((all_total_R_dense.shape[-1],))
    for i in range(all_total_R_dense.shape[-1]):
        neuron_outs_dense[i] = np.mean(all_total_R_dense[..., i])
    print('neuron_outs_dense: ', neuron_outs_dense.shape)

    num_relevant_neurons_dense = round(num_pro * len(neuron_outs_dense))  # int向下取整，ceil向上取整，round四舍五入
    print("num_relevant_neurons_dense: ", num_relevant_neurons_dense)

    #  根据proportion取top k important neurons
    top_k_neuron_indexes_dense = (
    np.argsort(neuron_outs_dense, axis=None)[-num_relevant_neurons_dense:len(neuron_outs_dense)])
    print('top_k_neuron_indexes_dense: ', top_k_neuron_indexes_dense)

    total_topk_neuron_idx = {}
    print('model.layers: ', model.layers[22].name)
    total_topk_neuron_idx[model.layers[22].name] = top_k_neuron_indexes_dense
    return total_topk_neuron_idx

def get_total_relevance_composition(model, X_train, Y_train):

    analyzer = innvestigate.create_analyzer("lrp.epsilon", model, reverse_keep_tensors=True)
    X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model, X_train, Y_train)
    print("X_train_corr shape: ", X_train_corr.shape)  # (88880, 28, 28, 1)

    # activation_5 and activation_6
    analysis = analyzer.analyze(X_train_corr[0:1])
    for k, v in analyzer._reversed_tensors:
        print("k: ", k)
        print("v.shape",v.shape)
        print('-------------------------')

    # block_conv5 + fc2 + fc3 + fc4
    all_total_R_act2 = np.zeros(analyzer._reversed_tensors[2][1].shape)
    all_total_R_act3 = np.zeros(analyzer._reversed_tensors[3][1].shape)
    all_total_R_act4 = np.zeros(analyzer._reversed_tensors[4][1].shape)
    all_total_R_act5 = np.zeros(analyzer._reversed_tensors[5][1].shape)
    all_total_R_act7 = np.zeros(analyzer._reversed_tensors[7][1].shape)
    all_total_R_act8 = np.zeros(analyzer._reversed_tensors[8][1].shape)

    for inp in range(len(X_train_corr)):
        print('inp: ', inp)
        analysis = analyzer.analyze(X_train_corr[inp * 1:inp * 1 + 1])
        relevance_act2 = analyzer._reversed_tensors[2][1]
        relevance_act3 = analyzer._reversed_tensors[3][1]
        relevance_act4 = analyzer._reversed_tensors[4][1]
        relevance_act5 = analyzer._reversed_tensors[5][1]
        relevance_act7 = analyzer._reversed_tensors[7][1]
        relevance_act8 = analyzer._reversed_tensors[8][1]

        all_total_R_act2 += relevance_act2
        all_total_R_act3 += relevance_act3
        all_total_R_act4 += relevance_act4
        all_total_R_act5 += relevance_act5
        all_total_R_act7 += relevance_act7
        all_total_R_act8 += relevance_act8

    np.save(path + 'lenet5_act2.npy', all_total_R_act2)
    np.save(path + 'lenet5_act3.npy', all_total_R_act3)
    np.save(path + 'lenet5_act4.npy', all_total_R_act4)
    np.save(path + 'lenet5_act5.npy', all_total_R_act5)
    np.save(path + 'lenet5_act7.npy', all_total_R_act7)
    np.save(path + 'lenet5_act8.npy', all_total_R_act8)


def get_topk_neurons_composition(model, num_pro):
    all_total_R_act2 = np.load(path + 'lenet5_act2.npy')
    all_total_R_act3 = np.load(path + 'lenet5_act3.npy')
    all_total_R_act4 = np.load(path + 'lenet5_act4.npy')
    all_total_R_act5 = np.load(path + 'lenet5_act5.npy')
    all_total_R_act7 = np.load(path + 'lenet5_act7.npy')
    all_total_R_act8 = np.load(path + 'lenet5_act8.npy')

    total_topk_neuron_idx = {}

    neuron_outs_act2 = np.zeros((all_total_R_act2.shape[-1],))
    for i in range(all_total_R_act2.shape[-1]):
        neuron_outs_act2[i] = np.mean(all_total_R_act2[..., i])
    num_relevant_neurons_act2 = round(num_pro * len(neuron_outs_act2))

    neuron_outs_act3 = np.zeros((all_total_R_act3.shape[-1],))
    for i in range(all_total_R_act3.shape[-1]):
        neuron_outs_act3[i] = np.mean(all_total_R_act3[..., i])
    num_relevant_neurons_act3 = round(num_pro * len(neuron_outs_act3))

    neuron_outs_act4 = np.zeros((all_total_R_act4.shape[-1],))
    for i in range(all_total_R_act4.shape[-1]):
        neuron_outs_act4[i] = np.mean(all_total_R_act4[..., i])
    num_relevant_neurons_act4 = round(num_pro * len(neuron_outs_act4))

    neuron_outs_act5 = np.zeros((all_total_R_act5.shape[-1],))
    for i in range(all_total_R_act5.shape[-1]):
        neuron_outs_act5[i] = np.mean(all_total_R_act5[..., i])
    num_relevant_neurons_act5 = round(num_pro * len(neuron_outs_act5))

    neuron_outs_act7 = np.zeros((all_total_R_act7.shape[-1],))
    for i in range(all_total_R_act7.shape[-1]):
        neuron_outs_act7[i] = np.mean(all_total_R_act7[..., i])
    num_relevant_neurons_act7 = round(num_pro * len(neuron_outs_act7))

    neuron_outs_act8 = np.zeros((all_total_R_act8.shape[-1],))
    for i in range(all_total_R_act8.shape[-1]):
        neuron_outs_act8[i] = np.mean(all_total_R_act8[..., i])
    num_relevant_neurons_act8 = round(num_pro * len(neuron_outs_act8))

    # 在python3.6里是activation5对应的是model.layers[12].name
    # 在python3.6里是activation6对应的是model.layers[14].name
    # 但是使用python3.8跑的sadl_improve文件，所以应该用11和13

    #  根据proportion取top k important neurons
    top_k_neuron_indexes_act2 = (
    np.argsort(neuron_outs_act2, axis=None)[-num_relevant_neurons_act2:len(neuron_outs_act2)])
    total_topk_neuron_idx[model.layers[1].name] = top_k_neuron_indexes_act2

    top_k_neuron_indexes_act3 = (
    np.argsort(neuron_outs_act3, axis=None)[-num_relevant_neurons_act3:len(neuron_outs_act3)])
    total_topk_neuron_idx[model.layers[2].name] = top_k_neuron_indexes_act3

    top_k_neuron_indexes_act4 = (
        np.argsort(neuron_outs_act4, axis=None)[-num_relevant_neurons_act4:len(neuron_outs_act4)])
    total_topk_neuron_idx[model.layers[3].name] = top_k_neuron_indexes_act4

    top_k_neuron_indexes_act5 = (
        np.argsort(neuron_outs_act5, axis=None)[-num_relevant_neurons_act5:len(neuron_outs_act5)])
    total_topk_neuron_idx[model.layers[4].name] = top_k_neuron_indexes_act5

    top_k_neuron_indexes_act7 = (
        np.argsort(neuron_outs_act7, axis=None)[-num_relevant_neurons_act7:len(neuron_outs_act7)])
    total_topk_neuron_idx[model.layers[6].name] = top_k_neuron_indexes_act7

    top_k_neuron_indexes_act8 = (
        np.argsort(neuron_outs_act8, axis=None)[-num_relevant_neurons_act8:len(neuron_outs_act8)])
    total_topk_neuron_idx[model.layers[7].name] = top_k_neuron_indexes_act8

    print('total_topk_neuron_idx: ', total_topk_neuron_idx)
    return total_topk_neuron_idx


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
    model = innvestigate.utils.model_wo_softmax(model)

    get_total_relevance(model, X_train, Y_train)
    # num_pro = 1
    # total_topk_neuron_idx = get_topk_neurons_composition(model, num_pro)

"""
print("(NodeID, TensorID) - min - max value")
for k, v in analyzer._reversed_tensors:

    print(v.shape)
    print(v)
    print('K: ', k)
    print(k, np.min(v), np.max(v))
    print('------------------------------------')


total_topk_neuron_idx:  {'block1_conv1': array([3, 4, 1, 2, 8, 0], dtype=int64), 'block1_pool1': array([3, 4, 1, 2, 8, 0], dtype=int64), 'block2_conv1': array([14,  4,  1, 12, 11,  2,  8,  9,  0, 18,  7,  3, 10,  6,  8, 13],
      dtype=int64), 'block2_pool1': array([14,  4,  1, 12, 11,  2,  8,  9,  0, 18,  7,  3, 10,  6,  8, 13],
      dtype=int64), 'fc1': array([34, 48, 88,  2, 60, 80,  0,  7, 18, 40, 82, 73, 31, 11, 33, 49, 74,
       36, 87, 82, 28, 21, 26, 14, 39, 32, 30, 72,  4,  8, 68,  6, 44, 78,
       43, 83, 23, 68, 17, 22, 64, 67, 83, 77, 29, 20, 62, 71,  8,  3, 38,
       12, 66, 88, 69, 41, 38, 18, 47,  9, 61, 16, 79, 81,  1, 76, 37, 46,
       24, 89, 70, 63, 48, 80, 84, 19, 28, 10, 42, 86, 13, 78, 27, 81],
      dtype=int64)}


# activation8
neuron_outs_dense = np.zeros((all_total_R_dense.shape[-1],))
for i in range(all_total_R_dense.shape[-1]):
    neuron_outs_dense[i] = np.mean(all_total_R_dense[..., i])
print('neuron_outs_dense: ', neuron_outs_dense.shape)

# activation6

neuron_outs_act6 = np.zeros((all_total_R_act6.shape[-1],))
for i in range(all_total_R_act6.shape[-1]):
    neuron_outs_act6[i] = np.mean(all_total_R_act6[..., i])
print('neuron_outs_act6: ', neuron_outs_act6.shape)

num_relevant_neurons_dense = round(num_pro * len(neuron_outs_dense))  # int向下取整，ceil向上取整，round四舍五入
num_relevant_neurons_act6 = round(num_pro * len(neuron_outs_act6))
print("num_relevant_neurons_dense: ", num_relevant_neurons_dense)
print("num_relevant_neurons_act6: ", num_relevant_neurons_act6)

#  根据proportion取top k important neurons
top_k_neuron_indexes_dense = (np.argsort(neuron_outs_dense, axis=None)[-num_relevant_neurons_dense:len(neuron_outs_dense)])
print('top_k_neuron_indexes_dense: ', top_k_neuron_indexes_dense)

top_k_neuron_indexes_act6 = (np.argsort(neuron_outs_act6, axis=None)[-num_relevant_neurons_act6:len(neuron_outs_act6)])
print('top_k_neuron_indexes_act6: ', top_k_neuron_indexes_act6)

total_topk_neuron_idx = {}
total_topk_neuron_idx[model.layers[11].name] = top_k_neuron_indexes_dense
total_topk_neuron_idx[model.layers[13].name] = top_k_neuron_indexes_act6

print('total_topk_neuron_idx: ', total_topk_neuron_idx)
return total_topk_neuron_idx

"""