from utils import *
from sadl_variant.innvestigate_code import  get_topk_neurons_composition
from scipy.spatial.distance import cdist
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import load_model, Model
from utils import *
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf
from multiprocessing import Pool
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
experiment_folder = "E:/githubAwesomeCode/1DLTesting/improve_DLtesting/sadl_variant/"
pre_lid_loc = experiment_folder + "/inputs_generation/cifar10_LID_act56_corr_other_eachlabel_20_14000_2600.npy"
pre_dsa_loc = experiment_folder + "cifar10/cifar10_dsa_11_13.npy"
pre_lsa_loc = experiment_folder + "cifar10/cifar10_lsa_11_13.npy"
pre_comb_loc = experiment_folder + "combination_values/2600/combination_normalized_right_14000_2600_0.5.npy"

train_ats_loc = experiment_folder + "gen_inputs/ats/cifar10_train_ats.npy"
train_pred_loc = experiment_folder + "gen_inputs/ats/cifar10_train_pred.npy"
train_ats_lid_loc = experiment_folder + "sadl_importance/cifar10_train_activation_5_activation_6_ats.npy"
train_pred_lid_loc = experiment_folder + "sadl_importance/cifar10_train_pred.npy"

gini_sum = experiment_folder + "/inputs_generation/gini_sum.npy"

is_classification = True; num_proc = 10
num_classes = 10
var_threshold = 1e-5


# 获取测试输入计算出的LID/DSA/LSA/combination值
def get_previous_values(idx, value):

    if value == 'lid':
        all_values = np.load(pre_lid_loc)
    if value =='dsa':
        all_values = np.load(pre_dsa_loc)
    if value == 'lsa':
        all_values = np.load(pre_lsa_loc)
    if value == 'comb':
        all_values = np.load(pre_comb_loc)

    idx_value = all_values[idx]
    return all_values, idx_value

def seed_filtering(model, seeds_index, X_test, Y_test):
    X_corr = []; Y_corr = []
    seeds_filter = []
    seeds_num = 30

    preds = model.predict(X_test[seeds_index])
    for idx, pred in enumerate(preds):
        # Y_test 里面的index不能写错
        if np.argmax(pred) == np.argmax(Y_test[seeds_index[idx]]):
            X_corr.append(X_test[idx])
            Y_corr.append(X_test[idx])
            seeds_filter.append(seeds_index[idx])

    seeds_filter = seeds_filter[0:seeds_num]
    print('seeds_filter: ', seeds_filter)

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

    num = 200  # 种子样本的数量为50
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


def get_lids_other(target_ats, train_ats, batch_size, layer_num, k, class_matrix, target_pred, all_idx):

    all_lids=[]; all_lids_normal=[]

    at = target_ats[-1]
    label = target_pred[-1].argmax(axis=-1)  # get the label for one test input
    label_train_ats = train_ats[class_matrix[label]]  # training inputs has the same label with the test input
    label_other_ats=  train_ats[list(set(all_idx) - set(class_matrix[label]))] # training inputs has the different labels with the test input

    print('label_train_ats: ', label_train_ats.shape)
    print('label_train_ats: ', label_other_ats.shape)

    #same label
    n_batches = int(np.ceil(label_train_ats.shape[0] / float(batch_size)))
    lids_adv = []
    for i_batch in range(n_batches): # for one batch
        lid_batch_adv = estimate(i_batch, batch_size, layer_num, at, label_train_ats, k)  #一个batch中的lids
        lids_adv.append(lid_batch_adv)
    lids_adv_input=np.average(lids_adv)    #lids_adv_input=np.var(lids_adv)
    all_lids.append(lids_adv_input)

    #  different label
    normal_batches = int(np.ceil(label_other_ats.shape[0] / float(batch_size)))
    lids_normal = []
    for i_batch in range(normal_batches):  # for one batch
        lid_batch_normal = estimate(i_batch, batch_size, layer_num, at, label_other_ats, k)  # 一个batch中的lids
        lids_normal.append(lid_batch_normal)
    lids_normal_input = np.average(lids_normal)
    all_lids_normal.append(lids_normal_input)

    all_result =  np.true_divide( all_lids, all_lids_normal)
    return all_result

def estimate(i_batch, batch_size, layer_num, at, train_ats, k):

    start = i_batch * batch_size
    end = np.minimum(len(train_ats), (i_batch + 1) * batch_size)
    n_feed = end - start
    lid_batch_adv = np.zeros(shape=(n_feed, layer_num))  # LID(adv):[j,1], j表示某个batch中的第几个样本, 1 表示只有1层

    X_act = train_ats[start:end]
    X_adv_act = at.reshape(1, at.shape[0])
    lid_batch_adv  = (mle_single(X_act, X_adv_act, k))

    return lid_batch_adv

# lid of a single query point x
def mle_single(data, x, k):
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))

    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]

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

def get_gen_img_lids(model, gen_img):

    batch_size = 1; num_pros = 0.20;
    b_s = 14000; k = 2600

    total_topk_neuron_idx = get_topk_neurons_composition(model, num_pros)

    layer_names=[]
    for key in total_topk_neuron_idx.keys():
        layer_names.append(key)

    #X_test = np.append(X_test, gen_img, axis=0)
    X_test = np.asarray(gen_img)
    print('X_test: ', X_test.shape)

    train_ats = np.load(train_ats_lid_loc)
    train_pred = np.load(train_pred_lid_loc)

    target_ats, target_pred  = get_target_ats_lid(model, X_test, layer_names, batch_size, total_topk_neuron_idx)
    print('target_ats: ', target_ats.shape); print('target_pred: ', target_pred.shape)

    class_matrix, all_idx = get_class_matrix (train_pred)
    inputs_lid = get_lids_other(target_ats, train_ats, b_s, len(layer_names), k, class_matrix, target_pred, all_idx)
    return inputs_lid[0]


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
    layer_names = []

    lyr = [11, 13]  # activation_8:21; dense_2:20
    for ly_id in lyr:
        layer_names.append(model.layers[ly_id].name)

    train_ats = np.load(train_ats_loc)
    train_pred = np.load(train_pred_loc)
    print('train_ats: ', train_ats.shape)

    X_test = np.asarray(gen_img)
    print('X_test: ', X_test.shape)

    target_ats, target_pred = get_target_ats(model, X_test, layer_names, batch_size)
    print('target_ats: ', target_ats.shape)   # (1, 256)
    print('target_pred: ', target_pred.shape)

    # 计算dsa值
    # dsa = get_dsa_values(train_ats, train_pred, target_ats, target_pred)

    # 计算lsa值,由于LSA值过大，容易导致图像的perturbation过大，所以不用该方法生成测试数据;并且LSA的capture surprise的效果最差
    # lsa = get_lsa_values(train_ats, train_pred, target_ats, target_pred)

    # 计算combination值，combination是LID与DSA的融合，
    comb_dsa = get_dsa_values(train_ats, train_pred, target_ats, target_pred)
    print('comb_dsa: ', comb_dsa)

    # comb_lid = get_gen_img_lids(model, gen_img)
    # print('comb_lid: ', comb_lid)
    # comb =  comb_dsa + comb_lid
    # print('comb: ', comb)

    return comb_dsa


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

if __name__ == "__main__":
    # load the data==============:
    dataset='cifar10'
    X_train, Y_train, X_test, Y_test = load_CIFAR()  # 在utils中修改

    # set the model====================
    model_path = r'E:/githubAwesomeCode/1DLTesting/improve_DLtesting/neural_networks/model_cifar'
    model_name = (model_path).split('/')[-1]
    print(model_name)
    model = load_model(model_path + '.h5')

    lr = 0.1
    total_sets = []
    seeds_index, seeds_data, seeds_label = seed_selection(4, X_test, Y_test)  # seeds_index 数据所在下标
    print('seeds_index: ', seeds_index)  #[2986]
    # seed_filtering 过滤测试数据集中被模型错误预测的样本，仅保留correctlty classified样本
    seeds_filter = seed_filtering(model, seeds_index, X_test, Y_test)

    # seeds_index = seed_deepgini(model, X_test)
    # seeds_filter = seed_filtering(model, seeds_index[0:200], X_test, Y_test)

    count = 0
    for idx in seeds_filter:
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

            for _ in range(3):   # range（3）即：从0到3，不包含3，即0,1,2
                gen_img = gen_img + dl_di * lr * (random.random() + 0.5)
                gen_img = tf.clip_by_value(gen_img, clip_value_min=0, clip_value_max=1)

                # 计算当前的dsa/DSA/LSA/COMB值
                dsa = get_gen_img_other(model, gen_img)
                print('current dsa: ', dsa)
                distance = np.linalg.norm(gen_img.numpy() - orig_img) / orig_norm

                #line 15-27
                if dsa > dsaMAX and distance < 0.5:
                    dsaMAX = dsa
                    img_list.append(tf.identity(gen_img))
                    print('append to the img_list')

                gen_index = np.argmax(model(gen_img)[0])
                print('orig_index: ', orig_index)
                print('gen_index: ', gen_index)
                if gen_index != orig_index:
                    print('=======adv======')
                    total_sets.append((dsa, gen_img.numpy(), Y_test[idx]))


    print('total_sets: ', total_sets)
    dsas = np.array([item[0] for item in total_sets])
    advs = np.array([item[1].reshape(32, 32, 3) for item in total_sets])
    labels = np.array([item[2] for item in total_sets])

    np.savez('./results0318/comb_dsaFuzz_seedsfilter_combSeeds_30_3.npz', advs=advs, labels=labels, dsas =dsas)


