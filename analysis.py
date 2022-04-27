import pandas as pd
import numpy as np
import os
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

EPSILON = 0.00000001
categories = ['BookOfProverb', 'TaoTeChing', 'BookOfEcclesiastes', 'Buddhism', 'BookOfWisdom', 'YogaSutra', 'BookOfEccleasiasticus', 'Upanishad']

def parse_hist_data(data_dir):
    words = None
    instances = []
    labels = []
    with open(os.path.join(data_dir, 'AllBooks_baseline_DTM_Labelled.csv'), 'r') as f:
        for line_id, line in enumerate(f.readlines()):
            if(line_id == 0):
                words = line.strip().split(',')[1:]
            else:
                hist = [int(x) for x in line.strip().split(',')[1:]]
                instances.append(hist)
                labels.append(categories.index(line.strip().split(',')[0].split('_')[0]))
    return words, np.array(instances), np.array(labels)

def pos_exp(train_x, train_y, val_x, val_y):
    train_x = normalize(train_x.astype(np.float64))
    val_x = normalize(val_x.astype(np.float64))

    cate_num = len(categories)
    lw = train_x.shape[1]
    w = np.zeros((cate_num, lw), dtype=np.float64)

    # collect category-wise data hist
    for i in range(cate_num):
        index = (train_y == i)
        w[i, :] = np.mean(train_x[index, :], axis=0)

    # normalize word appearence hist
    for j in range(lw):
        w[:, j] /= np.sum(w[:, j]) + EPSILON

    logits = val_x @ np.transpose(w) #pos_w - val_x @ neg_w
    pred = np.argmax(logits, axis=1)
    acc = np.sum(pred==val_y)*100./len(pred)
    return acc

def neg_exp(train_x, train_y, val_x, val_y):
    train_x = normalize(train_x.astype(np.float64))
    val_x = normalize(val_x.astype(np.float64))

    cate_num = len(categories)
    lw = train_x.shape[1]
    w = np.zeros((cate_num, lw), dtype=np.float64)

    # collect category-wise data hist
    for i in range(cate_num):
        index = (train_y == i)
        w[i, :] = np.mean(train_x[index, :], axis=0)

    # normalize word appearence hist
    for j in range(lw):
        w[:, j] /= np.sum(w[:, j]) + EPSILON
    w = 1. - w
    # for j in range(lw):
    #     w[:, j] /= np.sum(w[:, j]) + EPSILON

    logits = val_x @ np.transpose(w) #pos_w - val_x @ neg_w
    pred = np.argmin(logits, axis=1)
    acc = np.sum(pred==val_y)*100./len(pred)
    return acc

def k_centers_exp(train_x, train_y, val_x, val_y):
    train_x = normalize(train_x.astype(np.float64))
    val_x = normalize(val_x.astype(np.float64))
    cate_num = len(categories)
    lw = train_x.shape[1]
    w = np.zeros((cate_num, lw), dtype=np.float64)

    # collect category-wise data hist
    for i in range(cate_num):
        index = (train_y == i)
        w[i, :] = np.mean(train_x[index, :], axis=0)

    pred = np.zeros((len(val_x),), dtype=val_y.dtype)
    for i in range(len(val_x)):
        dists = val_x[i, :] - w
        dists = np.sum(np.square(dists), axis=1)
        pred[i] = np.argmin(dists) 

    return np.sum(pred==val_y)*100./len(pred)

def normalize(data, axis=1):
    assert(axis in [0, 1])
    # normalize instance hist
    if(axis == 1):
        for i in range(len(data)):
            data[i, :] /= np.sum(data[i, :]) + EPSILON
    else:
        for i in range(data.shape[1]):
            data[:, i] /= np.sum(data[:, i]) + EPSILON
    return data

def linear_reg_exp(train_x, train_y, val_x, val_y):
    train_x = normalize(train_x.astype(np.float64))
    val_x = normalize(val_x.astype(np.float64))
    
    reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    reg.fit(train_x, train_y)
    pred = np.clip(np.around(reg.predict(val_x)), 0, len(categories)-1)
    return np.sum(pred==val_y)*100./len(pred)

def SVC_exp(train_x, train_y, val_x, val_y):
    train_x = normalize(train_x.astype(np.float64))
    val_x = normalize(val_x.astype(np.float64))
    
    dt = SVC()
    dt.fit(train_x, train_y)
    preds = dt.predict(val_x)
    return np.sum(preds==val_y)*100./len(preds)

def RandomForest_exp(train_x, train_y, val_x, val_y):
    train_x = normalize(train_x.astype(np.float64))
    val_x = normalize(val_x.astype(np.float64))

    dt = RandomForestClassifier()
    dt.fit(train_x, train_y)
    preds = dt.predict(val_x)
    return np.sum(preds==val_y)*100./len(preds)

def DecisionTree_exp(train_x, train_y, val_x, val_y):
    train_x = normalize(train_x.astype(np.float64))
    val_x = normalize(val_x.astype(np.float64))

    dt = DecisionTreeClassifier()
    dt.fit(train_x, train_y)
    preds = dt.predict(val_x)
    return np.sum(preds==val_y)*100./len(preds)

def arr_max(a1, a2):
    assert(len(a1.flatten()) == len(a2.flatten()))
    return np.max(np.vstack((a1.reshape(1, -1), a2.reshape(1, -1))), axis=0)

def gen_mask(w):
    mask = w.copy()
    lw, cate_num = mask.shape
    # compare with max
    for j in range(cate_num):
        if(j == 0):
            comp = np.max(mask[:, 1:])
        elif(j == cate_num - 1):
            comp = np.max(mask[:, :-1])
        else:
            comp = arr_max(np.max(mask[:, :j]), np.max(mask[:, j+1:]))
        mask[:, j] = mask[:, j] - comp

    # compare with mean
    temp = np.sum(mask, axis=1)
    for j in range(cate_num):
        mask[:, j] = mask[:, j] - (temp - mask[:, j]) / (cate_num - 1)
    mask = np.abs(mask) < 0.05
    return mask

def get_top_words(instas, top_k=10):
    hist = np.sum(instas, axis=0)
    ind = hist.argsort()[-top_k:][::-1]
    return ind

if __name__ == '__main__':
    from tqdm import tqdm
    import statistics
    import math
    words, insts, labels = parse_hist_data('data')
    print("Got %d instances of %d categories" % (len(insts), len(categories)))
    print(categories)

    data = np.hstack((insts, labels.reshape(-1,1)))

    N = len(data)
    folds = 100
    pos_accs = []
    LR_accs = []
    neg_accs = []
    RF_accs = []
    DT_accs = []
    SVC_accs = []
    kc_accs = []
    for _ in tqdm(range(folds)):
        np.random.shuffle(data)
        train_num = N * 4 // 5
        train_x = data[:train_num, :-1]
        val_x = data[train_num:, :-1]
        train_y = data[:train_num, -1]
        val_y = data[train_num:, -1]
        # pos / neg / RandomForest / DecisionTree / SVC
        pos_accs.append(pos_exp(train_x, train_y, val_x, val_y))
        LR_accs.append(linear_reg_exp(train_x, train_y, val_x, val_y))
        RF_accs.append(RandomForest_exp(train_x, train_y, val_x, val_y))
        DT_accs.append(DecisionTree_exp(train_x, train_y, val_x, val_y))
        SVC_accs.append(SVC_exp(train_x, train_y, val_x, val_y))
        kc_accs.append(k_centers_exp(train_x, train_y, val_x, val_y))

    print("pos Mean=%.2f stdev=%.2f" % (sum(pos_accs)/len(pos_accs), statistics.stdev(pos_accs)))
    print("LR Mean=%.2f stdev=%.2f" % (sum(LR_accs)/len(LR_accs), statistics.stdev(LR_accs)))
    print("RF Mean=%.2f stdev=%.2f" % (sum(RF_accs)/len(RF_accs), statistics.stdev(RF_accs)))
    print("DT Mean=%.2f stdev=%.2f" % (sum(DT_accs)/len(DT_accs), statistics.stdev(DT_accs)))
    print("SVC Mean=%.2f stdev=%.2f" % (sum(SVC_accs)/len(SVC_accs), statistics.stdev(SVC_accs)))
    print("K centers Mean=%.2f stdev=%.2f" % (sum(kc_accs)/len(kc_accs), statistics.stdev(kc_accs)))
