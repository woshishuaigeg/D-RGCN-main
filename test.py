# -*- coding:utf-8 -*-
import time
import tensorflow as tf
from utils import *
from sklearn.model_selection import RepeatedKFold
import numpy as np
from ClassifierOutput import *
import pandas as pd
import os
from imblearn.combine import SMOTETomek, SMOTEENN  # smotune尝试使用
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE, ADASYN, SMOTE

# Set random seed
seed = 222
np.random.seed(seed)
# tf.set_random_seed(seed)
tf.random.set_seed(seed)


def run_evaluation(X_train, y_train, X_test, y_test):
    t = time.time()
    # If the proportion of positive samples in the training set exceeds 40%, it will be balanced
    #  teep

    if label_sum(y_train) > (int(len(y_train) * 0.4)):
        print("The training data does not need balance.")
        X_resampled, y_resampled = X_train, y_train
    else:
        X_resampled, y_resampled = SMOTETomek().fit_resample(X_train, y_train)
        # X_resampled, y_resampled = X_train, y_train

    # X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X_train, y_train)
    # shuffle the data and labels
    state = np.random.get_state()
    np.random.shuffle(X_resampled)
    np.random.set_state(state)
    np.random.shuffle(y_resampled)

    # training classifier
    # _, _, precision, recall, fmeasure, _, _ = \
    #     classifier_output('MLP', X_resampled, y_resampled, X_test, y_test,
    #                       grid_sear=True)  # False is only for debugging.

    # teep
    predprob_auc, predprob, precision, recall, fmeasure, auc, mcc, accuracy = \
        classifier_output('MLP', X_resampled, y_resampled, X_test, y_test,
                          grid_sear=True)  # False is only for debugging.

    print("precision=", "{:.5f}".format(precision),
          "recall=", "{:.5f}".format(recall),
          "f-measure=", "{:.5f}".format(fmeasure),
          "auc=", "{:.5f}".format(auc),
          "accuracy=", "{:.5f}".format(accuracy),
          "mcc=", "{:.5f}".format(mcc),
          "time=", "{:.5f}".format(time.time() - t))
    return precision, recall, fmeasure, auc, accuracy, mcc


def load_within_train_test(baseURL, project, mode):
    F1_list = []
    precision_list = []
    recall_list = []
    AUC_list = []
    accuracy_list = []
    mcc_list = []

    if mode == 'origin':
        # Traditional Static Code Metric(TCM)：20 dimension
        file = pd.read_csv(baseURL + project + "/Process-Binary.csv", header=0, index_col=False)
        X = np.array(file.iloc[:, 1:-1])

    elif mode == 'gcn':
        # GCN emb：32 dimension
        file = pd.read_csv(baseURL + project + "/gcn_emb_dgcn.emd", sep=" ", header=None, index_col=False)
        X1 = np.array(file.iloc[:, 1:-1])

    origin_train_data = pd.read_csv(baseURL + project + "/Process-Binary.csv", header=0,
                                    index_col=False)
    y = np.array(origin_train_data['bug'])
    # np.set_printoptions(threshold=np.inf)
    # print(y1)
    # print("go")
    # y = np.resize(y1, len(y1) * 2)
    # print(y)
    # file1 = pd.read_csv(baseURL + "/Camel" + "/gcn_emb.emd", sep=" ", header=None, index_col=False)
    # X2 = np.array(file1.iloc[:, 1:-1])
    #
    # origin_train_data1 = pd.read_csv(baseURL + "/Camel" + "/Process-Binary.csv", header=0,
    #                                 index_col=False)
    # y2 = np.array(origin_train_data1['bug'])

    exp_cursor = 1
    kf = RepeatedKFold(n_splits=5, n_repeats=5)  # We can modify n_repeats when debugging.
    for train_index, test_index in kf.split(X1, y):
        X1_train, X1_test = X1[train_index], X1[test_index]
        y1_train, y1_test = y[train_index], y[test_index]

        # for train_index, test_index in kf.split(X2, y2):
        #     X2_train, X2_test = X2[train_index], X2[test_index]
        #     y2_train, y2_test = y2[train_index], y2[test_index]

        precision, recall, fmeasure, auc, accuracy, mcc = run_evaluation(X1_train, y1_train, X1_test, y1_test)

        F1_list.append(fmeasure)
        precision_list.append(precision)
        recall_list.append(recall)
        AUC_list.append(auc)
        accuracy_list.append(accuracy)
        mcc_list.append(mcc)

        exp_cursor = exp_cursor + 1

    avg = []
    avg.append(average_value(F1_list))
    avg.append(average_value(precision_list))
    avg.append(average_value(recall_list))
    avg.append(average_value(AUC_list))
    avg.append(average_value(accuracy_list))
    avg.append(average_value(mcc_list))

    name = ['F1', 'precision', 'recall', 'AUC', 'Accuracy', 'MCC']
    results = []
    results.append(F1_list)
    results.append(precision_list)
    results.append(recall_list)
    results.append(AUC_list)
    results.append(accuracy_list)
    results.append(mcc_list)
    df = pd.DataFrame(data=results)
    df.index = name
    df.insert(0, 'avg', avg)
    df.to_csv('./result-dgcn/' + project + '/' + mode + '.csv')


# Assuming you have multiple sets of data to save


# loop eight projects
baseURL = "./data/"
# projects = ['velocity1_6_1']
# projects = ['Ant', 'Camel', 'Ivy','jEdit', 'Lucene', 'Poi', 'Velocity', 'Xalan']
# projects = ['Camel', 'jEdit', 'Lucene', 'Poi', 'Velocity', 'Xalan']
# projects = ['Ant', "ant1_3", "ant1_4", "ant1_5", "ant1_6", 'Camel', 'Ivy', 'jEdit', 'Lucene', 'Poi', 'Velocity',
# 'Xalan']
# projects = ["ant1_5", "ant1_6", "ant1_7"]
# projects = ['Ivy']
# projects = ['Velocity']
projects = ['xerces1_2']
# projects = [
#     "log4j1_2",
#     "lucene2_0",
#     "lucene2_2",
#     "lucene2_4",
#     "poi2_0",
#     "poi2_5_1",
#     "poi3_0",
#     "velocity1_4",
#     "velocity1_5",
#     "velocity1_6_1",
#     "xalan2_4",
#     "xalan2_5",
#     "xalan2_6",
#     "xerces1_2",
#     "xerces1_3",
#     "xerces1_4"]

# projects = [
# "ivy1_4",
# "ivy2_0",
# "jedit3_2_1",
# "jedit4_0",
# "jedit4_1",
# "log4j1_0",
# "log4j1_1"]
# projects = [
#     "ivy1_4",
#     "ivy2_0",
#     "jedit3_2_1",
#     "jedit4_0",
#     "jedit4_1",
#     "log4j1_0",
#     "log4j1_1"]
# projects = [
#     # "ant1_5", "ant1_6", "ant1_7",
#     # "ivy1_4",
#     # "ivy2_0",
#     # "jedit3_2_1",
#     # "jedit4_0",
#     # "jedit4_1",
#     # "log4j1_0",
#     # "log4j1_1",
#     # "log4j1_2",
#     # "lucene2_0",
#     # "lucene2_2",
#     "lucene2_4",
#     "poi2_0",
#     "poi2_5_1",
#     "poi3_0",
#     "velocity1_4",
#     "velocity1_5",
#     "velocity1_6_1",
#     "xalan2_4",
#     "xalan2_5",
#     "xalan2_6",
#     "xerces1_2",
#     "xerces1_3",
#     "xerces1_4"]
for i in range(len(projects)):
    print(projects[i] + " Start!")
    # mode: origin, metric, vector, origin_metric, origin_vector, metric_vector, origin_metric_vector, gcn
    load_within_train_test(baseURL, projects[i], 'gcn')
