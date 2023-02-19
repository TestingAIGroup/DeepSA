
from keras.models import Model
from utils import *
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf
from multiprocessing import Pool
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
experiment_folder = "C:/exprement/lenet5/"

pre_dsa_loc = experiment_folder + "lenet5_dsa_ats_40.npy"
pre_lsa_loc = experiment_folder + "lenet5_lsa_ats_40.npy"

train_ats_loc = experiment_folder + "mnist_train_block1_conv1_block1_pool1_block2_conv1_block2_pool1_fc1_fc2_ats.npy"
train_pred_loc = experiment_folder + "mnist_train_pred.npy"



is_classification = True; num_proc = 1
num_classes = 10
var_threshold = 1e-5
path = "C:/exprement/lenet5/"


# 获取测试输入计算出的LID/DSA/LSA/combination值
def get_previous_values(idx, value):

    if value =='dsa':
        all_values = np.load(pre_dsa_loc)
    if value == 'lsa':
        all_values = np.load(pre_lsa_loc)

    idx_value = all_values[idx]
    return all_values, idx_value

def seed_filtering_each_label(model, seeds_index, X_test, Y_test):
    X_corr = []; Y_corr = []
    seeds_each_label = {}  # key为label，value为对应的dsa值高的样本所在的下标
    seeds_filter = []   # 二维list，每个lable各三个样本

    preds = model.predict(X_test[seeds_index])
    for idx, pred in enumerate(preds):
        # Y_test 里面的index不能写错
        pred_label = np.argmax(pred)
        if pred_label == np.argmax(Y_test[seeds_index[idx]]):
            X_corr.append(X_test[idx])
            Y_corr.append(X_test[idx])
            seeds_each_label.setdefault(pred_label,[]).append(seeds_index[idx])

    for key in seeds_each_label:
        seeds_filter.append(seeds_each_label[key][:3])

    seeds_filter = [n for a in seeds_filter for n in a]
    print('seeds_filter: ', seeds_filter, len(seeds_filter))
    return seeds_filter

def seed_filtering(model, seeds_index, X_test, Y_test):
    X_corr = []; Y_corr = []
    seeds_filter = []
    seeds_num = 100

    preds = model.predict(X_test[seeds_index])
    for idx, pred in enumerate(preds):
        # Y_test 里面的index不能写错
        if np.argmax(pred) == np.argmax(Y_test[seeds_index[idx]]):
            X_corr.append(X_test[idx])
            Y_corr.append(X_test[idx])
            seeds_filter.append(seeds_index[idx])

    seeds_filter = seeds_filter[0:seeds_num]
    print('seeds_filter: ', seeds_filter, len(seeds_filter))

    return seeds_filter

def seed_deepgini(model, X_test):

    # 取gini不纯度高的
    all_gini=[]

    for idx in range(X_test.shape[0]):
        temp_img = X_test[[idx]]
        logits = model(temp_img)

        pro_sum = 0
        for pro in logits[0]:
            pro_sum = pro_sum + pro*pro
        t_gini = 1 - pro_sum

        all_gini.append(t_gini)

    gini_idx = np.argsort(all_gini)[::-1]
    print('all_gini: ', gini_idx)

    return gini_idx

def seed_dlregion(model, X_test):

    # 选取最大预测概率最低的test inputs作为seeds
    all_max_pro = []

    for idx in range(X_test.shape[0]):
        temp_img = X_test[[idx]]
        logits = model(temp_img)
        orig_index = np.argmax(logits[0])   #概率值最大的所在的index，也是对应的class label

        max_pro = logits[0][orig_index]    #最大概率值
        all_max_pro.append(max_pro)

    min_pro_idx = np.argsort(all_max_pro)
    # print('all_max_pro: ', min_pro_idx)   # 最大预测概率按照从小到大顺序排序

    return min_pro_idx

# lid 和 dsa值归一化
def max_min_normalization(dsa_values):
    max = np.max(dsa_values)
    min = np.min(dsa_values)
    norm = np.true_divide(dsa_values - min, max - min)

    return norm

def seed_selection(strategy, X_test, Y_test):
    # strategy=1，取test inputs中LID值最高的50个inputs作为种子
    idx = 0

    num = 300  # 种子样本的数量为50
    if strategy == 1:
        lid_values, _ = get_previous_values(idx, 'lid')
        descending_index = np.argsort(lid_values)[::-1]   # [1734 3173 8521 ... 6048 1313 7876]  全部数据
        seeds_data = X_test[descending_index[0:num], :, :, :]
        seeds_label = Y_test[descending_index[0:num], :]
        seeds_index = descending_index[0:num]   # 取种子数量个数据

        return seeds_index, seeds_data, seeds_label

    if strategy == 2:
        dsa_values, _ = get_previous_values(idx, 'dsa')
        descending_index = np.argsort(dsa_values)[::-1]   # [1734 3173 8521 ... 6048 1313 7876]  全部数据
        seeds_data = X_test[descending_index[0:num], :, :, :]
        seeds_label = Y_test[descending_index[0:num], :]
        seeds_index = descending_index[0:num]

    if strategy == 3:
        dsa_values, _ = get_previous_values(idx, 'lsa')
        descending_index = np.argsort(dsa_values)[::-1]  # [1734 3173 8521 ... 6048 1313 7876]  全部数据
        seeds_data = X_test[descending_index[0:num], :, :, :]
        seeds_label = Y_test[descending_index[0:num], :]
        seeds_index = descending_index[0:num]
        return seeds_index, seeds_data, seeds_label

    if strategy == 4:
        dsa_values, _ = get_previous_values(idx, 'dsa')
        lid_values, _ = get_previous_values(idx, 'lid')
        # 数值归一化后直接相加
        comb_values = max_min_normalization(dsa_values) + max_min_normalization(lid_values)

        descending_index = np.argsort(comb_values)[::-1]  # [1734 3173 8521 ... 6048 1313 7876]  全部数据
        seeds_data = X_test[descending_index[0:num], :, :, :]
        seeds_label = Y_test[descending_index[0:num], :]
        seeds_index = descending_index[0:num]

    return seeds_index, seeds_data, seeds_label

def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

def get_class_matrix(train_pred):
    class_matrix = {}  # 输入为训练数据集，模型的输出，每个类对应的数据下标
    all_idx = []

    for i, label in enumerate(train_pred):
        if label.argmax(axis=-1) not in class_matrix:  # label.argmax表示返回最大值的索引
            class_matrix[label.argmax(axis=-1)] = []
        class_matrix[label.argmax(axis=-1)].append(i)
        all_idx.append(i)
    return class_matrix, all_idx

def get_target_ats(model, X_test, layer_names, batch_size):
    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names], )

    if is_classification:
        p = Pool(num_proc)
        pred = model.predict(X_test, batch_size=batch_size, verbose=1)

        if len(layer_names) == 1:
            layer_outputs = [temp_model.predict(X_test, batch_size=batch_size, verbose=1)]
        else:
            layer_outputs = temp_model.predict(X_test, batch_size=batch_size, verbose=1)

        ats = None
        for layer_name, layer_output in zip(layer_names, layer_outputs):  # (1, 60000, 4, 4, 12)
            if layer_output[0].ndim == 3:
                layer_matrix = np.array(p.map(_aggr_output, [layer_output[i] for i in range(len(X_test))]))
            else:
                layer_matrix = np.array(layer_output)

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None

    return ats, pred

def get_target_ats_lid(model, X_test, layer_names,  batch_size, topk_neuron_idx):
    is_classification =True;  num_proc=10

    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )

    if is_classification:
        p = Pool(num_proc)
        pred = model.predict(X_test, batch_size=batch_size, verbose=1)

        if len(layer_names) == 1:
            layer_outputs = [temp_model.predict(X_test, batch_size=batch_size, verbose=1)]
        else:
            layer_outputs = temp_model.predict(X_test, batch_size=batch_size, verbose=1)

        ats = None

        for layer_name, layer_output in zip(layer_names, layer_outputs):  # (1, 60000, 4, 4, 12)
            if layer_output[0].ndim == 3:
                list_top_neuron_idx = topk_neuron_idx[layer_name]
                layer_matrix = np.array( p.map(_aggr_output, [layer_output[i][:, :, list_top_neuron_idx] for i in range(len(X_test))]) )
            else:
                list_top_neuron_idx = topk_neuron_idx[layer_name]
                layer_matrix = np.array(layer_output[:, list_top_neuron_idx])

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None

    return ats, pred

def get_lsa_values(train_ats, train_pred, target_ats, target_pred):
    class_matrix = {}
    is_classification =True

    if is_classification:
        for i, label in enumerate(train_pred):
            if label.argmax(axis=-1) not in class_matrix:
                class_matrix[label.argmax(axis=-1)] = []
            class_matrix[label.argmax(axis=-1)].append(i)
    print(class_matrix.keys())

    kdes, removed_cols = _get_kdes(train_ats, train_pred, class_matrix, is_classification, num_classes, var_threshold)
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
    print('lsa[0]: ', lsa[0])
    return lsa[0]

def _get_lsa(kde, at, removed_cols):
    refined_at = np.delete(at, removed_cols, axis=0)
    return np.asscalar(-kde.logpdf(np.transpose(refined_at)))

def _get_kdes(train_ats, train_pred, class_matrix, is_classification, num_classes, var_threshold):

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
            if refined_ats.shape[0] == 0:
                print("ats were removed by threshold {}".format(var_threshold))
                break
            kdes[label] = gaussian_kde(refined_ats)

    # for regression models
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

    print("The number of removed columns: {}".format(len(removed_cols)))
    return kdes, removed_cols

def get_gen_img_other(model, gen_img):
    batch_size = 1
    num_pros = 0.40
    b_s = 14000
    k = 2600

    X_train_corr, Y_train_corr, _, _ = filter_correct_classifications(model, X_train, Y_train)
    total_topk_neuron_idx = get_topk_neurons_composition(model, num_pros)

    layer_names = []
    for key in total_topk_neuron_idx.keys():
        layer_names.append(key)

    X_test = np.asarray(gen_img)
    print('X_test: ', X_test.shape)

    train_ats = np.load(train_ats_loc)
    train_pred = np.load(train_pred_loc)

    target_ats, target_pred = get_target_ats_lid(model, X_test, layer_names, batch_size, total_topk_neuron_idx)
    print('target_ats: ', target_ats.shape)
    print('target_pred: ', target_pred.shape)

    #计算dsa值
    dsa = get_dsa_values(train_ats, train_pred, target_ats, target_pred)

    # 计算lsa值,由于LSA值过大，容易导致图像的perturbation过大，所以不用该方法生成测试数据;并且LSA的capture surprise的效果最差
    # lsa = get_lsa_values(train_ats, train_pred, target_ats, target_pred)

    # 计算combination值，combination是LID与DSA的融合，
    # comb_dsa = get_dsa_values(train_ats, train_pred, target_ats, target_pred)
    # print('comb_dsa: ', comb_dsa)

    # comb_lid = get_gen_img_lids(model, gen_img)
    # print('comb_lid: ', comb_lid)
    # comb =  comb_dsa + comb_lid
    # print('comb: ', comb)

    return dsa

def find_closest_at(at, train_ats):
    #The closest distance between subject AT and training ATs.
    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])

def get_dsa_values(train_ats, train_pred, target_ats, target_pred):

    class_matrix = {}; all_idx = []
    for i, label in enumerate(train_pred):
        if label.argmax(axis=-1) not in class_matrix:  # label.argmax表示返回最大值的索引
            class_matrix[label.argmax(axis=-1)] = []
        class_matrix[label.argmax(axis=-1)].append(i)
        all_idx.append(i)

    dsa = []
    for i, at in enumerate(target_ats):
        label = target_pred[i].argmax(axis=-1)
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
        b_dist, _ = find_closest_at(a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))])
        dsa.append(a_dist / b_dist)

    print('dsa[0]: ', dsa[0])
    return dsa[0]


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


    return total_topk_neuron_idx


def gen_inputs(seeds_filter, X_test, model):

    for idx in seeds_filter:
        gen_img_list = []

        print('idx: ', idx)

        img_list = []
        tmp_img = X_test[[idx]]
        orig_img = tmp_img.copy()
        orig_norm = np.linalg.norm(orig_img)  # ord=None, 求整个矩阵元素平方和再开根号
        img_list.append(tf.identity(tmp_img))

        logits = model(tmp_img)   # 获取tmp_img的概率向量
        orig_index = np.argmax(logits[0])
        print('orig_index: ', orig_index)
        print('Y_test_label: ', Y_test[idx])   # 将种子数据的标签记录下来，存放到total_sets中。这里Y_test 与 orig_index不一致是因为模型预测错误。存放到total_sets中的应该是真实label

        target = keras.utils.to_categorical([orig_index], 10)
        label_top5 = np.argsort(logits[0])[-5:]

        dsaMAX = 0
        epoch = 0

        while len(img_list) > 0:

            gen_img = img_list.pop(0)

            gen_img = tf.Variable(gen_img)
            dsa = get_gen_img_other(model, gen_img)
            print('previous_dsa:', dsa)
            dsaMAX = dsa

            with tf.GradientTape(persistent=True) as g:
                logits = model(gen_img)
                other_losses = logits[0][label_top5[-2]] + logits[0][label_top5[-3]] + logits[0][label_top5[-4]] + \
                               logits[0][label_top5[-5]]
                obj = other_losses -  logits[0][orig_index] + dsa
                print('obj: ', obj)
                dl_di = g.gradient(obj, gen_img)
            del g

            for _ in range(5):   # range（3）即：从0到3，不包含3，即0,1,2
                gen_img = gen_img + dl_di * lr * (random.random() + 0.5)
                gen_img = tf.clip_by_value(gen_img, clip_value_min=0, clip_value_max=1)

                # 计算当前的dsa/DSA/LSA/COMB值
                dsa = get_gen_img_other(model, gen_img)
                print('current dsa: ', dsa)
                distance = np.linalg.norm(gen_img.numpy() - orig_img) / orig_norm

                gen_index = np.argmax(model(gen_img)[0])
                print('orig_index: ', orig_index)
                print('gen_index: ', gen_index)
                if gen_index != orig_index:
                    print('=======adv======')
                    total_sets.append((dsa, gen_img.numpy(), Y_test[idx]))

                #line 15-27
                if dsa > dsaMAX and distance < 0.5:
                    dsaMAX = dsa
                    if len(gen_img_list) < 25:
                        img_list.append(tf.identity(gen_img))
                        gen_img_list.append(gen_img)
                        print('append to the img_list')
                    else:
                        break
            print('len_gen_img_list', len(gen_img_list))
            if len(gen_img_list) >= 25 :
                break

    return  total_sets


if __name__ == "__main__":
    # load the data:
    dataset = 'mnist'
    X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)  # 在utils中修改
    img_rows, img_cols = 28, 28

    # set the model
    model_path = r'c:/exprement/lenet5/lenet5'
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

    lr = 0.1
    total_sets = []
    # 2:DSA, 3:LSA
    seeds_index, seeds_data, seeds_label = seed_selection(2, X_test, Y_test)  # seeds_index 数据所在下标
    print('seeds_index: ', seeds_index)  #[2986]
    # seed_filtering 过滤测试数据集中被模型错误预测的样本，仅保留correctlty classified样本
    seeds_filter = seed_filtering(model, seeds_index, X_test, Y_test)

    # seeds_index = seed_deepgini(model, X_test)
    # seeds_filter = seed_filtering(model, seeds_index[0:700], X_test, Y_test)
    # print('seeds_filter: ', seeds_filter, len(seeds_filter))

    # seeds_index = seed_dlregion(model, X_test)
    # seeds_filter = seed_filtering(model, seeds_index[0:600], X_test, Y_test)


    total_sets = gen_inputs(seeds_filter, X_test, model)

    print('total_sets: ', total_sets)
    dsas = np.array([item[0] for item in total_sets])
    advs = np.array([item[1].reshape(28, 28, 1) for item in total_sets])
    labels = np.array([item[2] for item in total_sets])
    gen_path = 'C:/exprement/lenet5/'
    np.savez(gen_path +'dsaFuzz_dsaSeeds_100_5.npz', advs=advs, labels=labels, dsas =dsas)


