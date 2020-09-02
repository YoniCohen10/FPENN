from os import listdir
from os.path import isfile, join

import pandas as pd
from keras_self_attention import SeqSelfAttention
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Input, concatenate, LSTM, Dropout, Bidirectional
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from time import time
from sklearn import tree as m_tree
import matplotlib.pyplot as plt
from sklearn import preprocessing
from gensim.models import Word2Vec
import itertools
from pathlib import Path
import keras.backend as K

import tensorflow as tf
import os

#
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";  # Do other imports now...
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# File Paths dictionary
IS_MULTICLASS = False
NUM_OF_CLASS = 1
NUM_OF_EPOCS = 2
MAX_FEATURES = 0.8
NEED_OVERSAMPLE = False

xgb_estimators = 15
xgb_depth = 4
rf_estimators = 15
rf_depth = 4


def get_instances_route(classifier, X, is_gbt, tree_dic=None):
    '''
    Takes in a GradientBoostingClassifier object (gbc) and a data frame (X).
    Returns a numpy array of dim (rows(X), num_estimators), where each row represents the set of terminal nodes
    that the record X[i] falls into across all estimators in the GBC.

    Note, each tree produces 2^max_depth terminal nodes. I append a prefix to the terminal node id in each incremental
    estimator so that I can use these as feature ids in other classifiers.
    '''
    if tree_dic is None:
        tree_dic = {}
        max_node = 0
        all_courses = []
        if is_gbt:
            for i, dt_i in enumerate(classifier.estimators_):
                instace_courses_indexes = []
                leaf_dic = {}
                prefix = max_node  # Must be an integer
                ind = 0
                instace_courses_enc = dt_i[0].decision_path(X, check_input=True).todense()
                for course in instace_courses_enc:
                    course = np.where(course == 1)[1]
                    instace_courses_indexes.append(course)
                for index_in_instace_courses, course in zip(range(len(instace_courses_indexes)),
                                                            instace_courses_indexes):
                    for node in course:
                        if node not in leaf_dic:
                            if node + prefix > max_node:
                                max_node = node + prefix
                            leaf_dic[node] = ind
                            ind = ind + 1
                    instace_courses_indexes[index_in_instace_courses] = [x + prefix for x in course]
                max_node += 1
                all_courses.append(instace_courses_indexes)
                for ind, value in leaf_dic.items():
                    leaf_dic[ind] = value + prefix
                tree_dic[i] = leaf_dic
        if not is_gbt:
            max_node = 0
            for i, dt_i in enumerate(classifier.estimators_):
                leaf_dic = {}
                prefix = max_node  # Must be an integer
                ind = 0
                instace_courses_enc = dt_i.decision_path(X, check_input=True).todense()
                instace_courses_indexes = []
                for course in instace_courses_enc:
                    course = np.where(course == 1)[1]
                    instace_courses_indexes.append(course)
                for index_in_instace_courses, course in zip(range(len(instace_courses_indexes)),
                                                            instace_courses_indexes):
                    for node in course:
                        if node not in leaf_dic:
                            if node + prefix > max_node:
                                max_node = node + prefix
                            leaf_dic[node] = ind
                            ind = ind + 1
                    instace_courses_indexes[index_in_instace_courses] = [x + prefix for x in course]
                max_node += 1
                all_courses.append(instace_courses_indexes)
                for ind, value in leaf_dic.items():
                    leaf_dic[ind] = value + prefix
                tree_dic[i] = leaf_dic
        return all_courses, max_node, tree_dic
    else:
        if is_gbt:
            all_courses = []
            for i, dt_i in enumerate(classifier.estimators_):
                instace_courses_enc = dt_i[0].decision_path(X, check_input=True).todense()
                instace_courses_indexes = []
                for course in instace_courses_enc:
                    course = np.where(course == 1)[1]
                    instace_courses_indexes.append(course)
                for index_in_instace_courses, course in zip(range(len(instace_courses_indexes)),
                                                            instace_courses_indexes):
                    for c_i, node in zip(range(len(course)), course):
                        instace_courses_indexes[index_in_instace_courses][c_i] = tree_dic[i][node]
                all_courses.append(instace_courses_indexes)
        if not is_gbt:
            all_courses = []
            for i, dt_i in enumerate(classifier.estimators_):
                instace_courses_enc = dt_i.decision_path(X, check_input=True).todense()
                instace_courses_indexes = []
                for course in instace_courses_enc:
                    course = np.where(course == 1)[1]
                    instace_courses_indexes.append(course)
                for index_in_instace_courses, course in zip(range(len(instace_courses_indexes)),
                                                            instace_courses_indexes):
                    for c_i, node in zip(range(len(course)), course):
                        instace_courses_indexes[index_in_instace_courses][c_i] = tree_dic[i][node]
                all_courses.append(instace_courses_indexes)
        return all_courses, 0, 0


def random_forest_classifier(features, target, num_est, mx_depth):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(n_estimators=num_est, max_depth=mx_depth, max_features=MAX_FEATURES)
    clf.fit(features, target)
    return clf


def xgboost_classifier(features, target, num_est, mx_depth):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = GradientBoostingClassifier(n_estimators=num_est, max_depth=mx_depth, max_features=MAX_FEATURES)
    clf.fit(features, target)
    return clf


def simple_nn_classifier(train_x, train_y):
    model = Sequential()
    model.add(Dense(256, input_dim=train_x.shape[1], activation='relu'))
    model.add(Dropout(0.3, input_shape=(256,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3, input_shape=(128,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(train_x, train_y, epochs=NUM_OF_EPOCS, verbose=0)
    return model


def run(i, INPUT_PATH):
    embedding_layer_size = 40
    INPUT_PATH = open(INPUT_PATH, 'r')
    dataset = pd.read_csv(INPUT_PATH)
    INPUT_PATH.close()
    train_x, test_x, train_y, test_y = train_test_split(dataset.iloc[:, 0:-1], dataset.iloc[:, -1], train_size=0.8,
                                                        random_state=123, stratify=dataset.iloc[:, -1])

    headers = train_x.columns.values
    if type(train_x) is not pd.DataFrame:
        train_x = pd.DataFrame(data=train_x, columns=headers)
        train_y = pd.Series(train_y)

    # # Train and Test dataset size details
    # print_dataset_details(test_x, test_y, train_x, train_y)

    # Define the rival model
    is_gbt = False
    trained_model = random_forest_classifier(train_x, train_y, rf_estimators, rf_depth)
    # trained_model_xgb = xgboost_classifier(train_x, train_y, xgb_estimators, xgb_depth)

    # Print rivals performances
    rival_auc_rf = roc_auc_score(test_y, trained_model.predict_proba(test_x)[:, 1:])
    # rival_auc_xgb = roc_auc_score(test_y, trained_model_xgb.predict_proba(test_x)[:, 1:])
    # print_rivals_performances(rival_auc_rf, rival_auc_xgb, test_x, test_y, train_x, train_y, trained_model,
    # trained_model_xgb)

    # Creating the NN basic parameters
    num_of_estimators_rf = trained_model.n_estimators
    leaf_rf, max_node_rf, tree_dic_rf = get_instances_route(trained_model, train_x, is_gbt)
    probs_rf = trained_model.predict_proba(train_x)
    vocab_size_rf = max_node_rf + 1

    # num_of_estimators_xgb = trained_model_xgb.n_estimators
    # leaf_xgb, max_node_xgb, tree_dic_xgb = get_instances_route(trained_model_xgb, train_x, not is_gbt)
    # probs_xgb = trained_model_xgb.predict_proba(train_x)
    # vocab_size_xgb = max_node_xgb + 1

    # Padding all seq to the same size
    max_len_rf = 0
    for t in leaf_rf:
        max_len_rf = max(max([len(x) for x in t]), max_len_rf)
    for tree_ind, tree in zip(range(len(leaf_rf)), leaf_rf):
        for course_ind, course in zip(range(len(tree)), tree):
            if len(leaf_rf[tree_ind][course_ind]) < max_len_rf:
                leaf_rf[tree_ind][course_ind] = [leaf_rf[tree_ind][course_ind][0]] * (
                        max_len_rf - len(leaf_rf[tree_ind][course_ind])) + list(leaf_rf[tree_ind][course_ind])

    # max_len_xgb = 0
    # for t in leaf_xgb:
    #     max_len_xgb = max(max([len(x) for x in t]), max_len_xgb)
    # for tree_ind, tree in zip(range(len(leaf_xgb)), leaf_xgb):
    #     for course_ind, course in zip(range(len(tree)), tree):
    #         if len(leaf_xgb[tree_ind][course_ind]) < max_len_xgb:
    #             leaf_xgb[tree_ind][course_ind] = [leaf_xgb[tree_ind][course_ind][0]] * (
    #                     max_len_xgb - len(leaf_xgb[tree_ind][course_ind])) + list(leaf_xgb[tree_ind][course_ind])
    # Train language model to RF nodes
    text_for_w2v_rf = list(itertools.chain.from_iterable(leaf_rf))
    for i in range(len(text_for_w2v_rf)):
        text_for_w2v_rf[i] = [str(x) for x in text_for_w2v_rf[i]]
    w2v_model_rf = Word2Vec(text_for_w2v_rf, size=embedding_layer_size, window=5, min_count=1, workers=4)

    # Train language model to XGB nodes
    # text_for_w2v_xgb = list(itertools.chain.from_iterable(leaf_xgb))
    # for i in range(len(text_for_w2v_xgb)):
    #     text_for_w2v_xgb[i] = [str(x) for x in text_for_w2v_xgb[i]]
    # w2v_model_xgb = Word2Vec(text_for_w2v_xgb, size=embedding_layer_size, window=5, min_count=1, workers=4)

    # Convert the nodes output to vectors using the LM
    course_input_rf = []
    for t_i, t in zip(range(len(leaf_rf)), leaf_rf):
        for c_i, course in zip(range(len(t)), t):
            l = list(map(lambda x: w2v_model_rf.wv[x], [str(x) for x in course]))
            course_input_rf.append(l)

    # course_input_xgb = []
    # for t_i, t in zip(range(len(leaf_xgb)), leaf_xgb):
    #     for c_i, course in zip(range(len(t)), t):
    #         l = list(map(lambda x: w2v_model_xgb.wv[x], [str(x) for x in course]))
    #         course_input_xgb.append(l)

    # RF
    main_input = Input(shape=(max_len_rf, embedding_layer_size), dtype='float32',
                       name='main_input')
    lstm_out = LSTM(32, return_sequences=True)(main_input)
    x = SeqSelfAttention(attention_activation='sigmoid')(lstm_out)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3, input_shape=(128,))(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3, input_shape=(256,))(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    auxiliary_output = Dense(NUM_OF_CLASS, activation='sigmoid', name='aux_output')(x)

    # XGB
    # auxiliary_input_second = Input(shape=(max_len_xgb, embedding_layer_size), dtype='float32',
    #                                name='auxiliary_input_second')
    # lstm_out_xgb = LSTM(32, return_sequences=True)(auxiliary_input_second)
    # y = SeqSelfAttention(attention_activation='sigmoid')(lstm_out_xgb)
    # y = Flatten()(y)
    # y = Dense(128, activation='relu')(y)
    # y = Dropout(0.3, input_shape=(1024,))(y)
    # y = Dense(128, activation='relu')(y)
    # y = Dense(128, activation='relu')(y)
    # y = Dropout(0.3, input_shape=(256,))(y)
    # y = Dense(32, activation='relu')(y)
    # y = Dense(32, activation='relu')(y)
    # auxiliary_output_sec = Dense(NUM_OF_CLASS, activation='sigmoid', name='auxiliary_output_sec')(y)

    # NN
    in_shape = train_x.shape[1] + NUM_OF_CLASS
    if not IS_MULTICLASS:
        in_shape = train_x.shape[
                       1] + NUM_OF_CLASS + 1  # raw data + prediction rf + prediction xgb + prob rf + prob xgb
    auxiliary_input = Input(shape=(in_shape,), name='aux_input')

    z = concatenate([x, auxiliary_input, auxiliary_output])

    z = Dense(256, activation='relu')(z)
    z = Dropout(0.3, input_shape=(256,))(z)
    z = Dense(256, activation='relu')(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.3, input_shape=(128,))(z)
    z = Dense(64, activation='relu')(z)
    main_output = Dense(NUM_OF_CLASS, activation='sigmoid', name='main_output')(z)
    model_deep_forest = Model(inputs=[main_input, auxiliary_input],
                              outputs=[main_output, auxiliary_output])
    if not IS_MULTICLASS:
        loss = "binary_crossentropy"
    else:
        loss = "categorical_crossentropy"
    model_deep_forest.compile(optimizer='adam',
                              loss={'main_output': loss, 'aux_output': loss},
                              metrics=['accuracy'])

    auxiliary_train_input_data = train_x.copy()
    if not IS_MULTICLASS:
        auxiliary_train_input_data["prob_0_rf"] = probs_rf[:, 0]
        auxiliary_train_input_data["prob_1_rf"] = probs_rf[:, 1]
        # auxiliary_train_input_data["prob_0_xgb"] = probs_xgb[:, 0]
        # auxiliary_train_input_data["prob_1_xgb"] = probs_xgb[:, 1]

        auxiliary_train_input_data_for_rivals = auxiliary_train_input_data.copy()
    # else:
    #     auxiliary_train_input_data = pd.concat(
    #         [auxiliary_train_input_data.reset_index(drop=True),
    #          pd.DataFrame(probs_rf, columns=list(map(lambda y: "prob_rf" + str(y), range(NUM_OF_CLASS))))],
    #         axis=1, sort=False)
    #     auxiliary_train_input_data = pd.concat(
    #         [auxiliary_train_input_data.reset_index(drop=True),
    #          pd.DataFrame(probs_xgb, columns=list(map(lambda y: "prob_xgb" + str(y), range(NUM_OF_CLASS))))],
    #         axis=1, sort=False)

    # CHECK if probs_rf helps
    # auxiliary_train_input_data["prob_0"] = np.random.normal(size=train_x.shape[0])
    # auxiliary_train_input_data["prob_1"] = np.random.normal(size=train_x.shape[0])

    # CHECK if leaf_rf helps
    # leaf_rf = np.random.randint(200, vocab_size_rf, size=(train_x.shape[0], num_of_estimators_rf))

    if IS_MULTICLASS:
        targets_train = to_categorical(train_y)
        targets_test = to_categorical(test_y)
    else:
        targets_train = train_y
        targets_test = test_y
        targets_train = pd.concat([targets_train] * rf_estimators, ignore_index=True)

    auxiliary_train_input_data = pd.concat([auxiliary_train_input_data] * rf_estimators, ignore_index=True)

    start = time()
    model_deep_forest.fit(
        {'main_input': np.array(course_input_rf), 'aux_input': auxiliary_train_input_data},
        {'main_output': targets_train, 'aux_output': targets_train},
        epochs=NUM_OF_EPOCS, batch_size=128, verbose=0)
    end = time()

    fe_elapse = end - start
    print("Total Time for epoc: " + str(fe_elapse))

    test_prob_rf = trained_model.predict_proba(test_x)
    test_res_rf = trained_model.predict(test_x)
    test_input_net = test_x.copy()

    # test_prob_xgb = trained_model_xgb.predict_proba(test_x)
    # test_res_xgb = trained_model_xgb.predict(test_x)

    if not IS_MULTICLASS:
        print("test res 0: " + str(list(test_res_rf).count(0)))
        print("test res 1: " + str(list(test_res_rf).count(1)))
        test_input_net["prob_0_rf"] = test_prob_rf[:, 0]
        test_input_net["prob_1_rf"] = test_prob_rf[:, 1]
        # test_input_net["prob_0_xgb"] = test_prob_xgb[:, 0]
        # test_input_net["prob_1_xgb"] = test_prob_xgb[:, 1]

        test_input_rivals = test_input_net.copy()
    else:
        test_input_net = pd.concat(
            [test_input_net.reset_index(drop=True),
             pd.DataFrame(test_prob_rf, columns=list(map(lambda x: "prob_rf_" + str(x), range(NUM_OF_CLASS))))],
            axis=1, sort=False)
        test_input_net = pd.concat(
            [test_input_net.reset_index(drop=True),
             pd.DataFrame(test_prob_xgb, columns=list(map(lambda x: "prob_xgb_" + str(x), range(NUM_OF_CLASS))))],
            axis=1, sort=False)

    # This is the test phase after we created the new model we evaluate it on the test set
    # print("The current fold is:" + str(i))

    # Prepare data to test phase
    rf_leaf_test = get_instances_route(trained_model, test_x, is_gbt, tree_dic_rf)[0]
    for tree_ind, tree in zip(range(len(rf_leaf_test)), rf_leaf_test):
        for course_ind, course in zip(range(len(tree)), tree):
            if len(rf_leaf_test[tree_ind][course_ind]) < max_len_rf:
                rf_leaf_test[tree_ind][course_ind] = [rf_leaf_test[tree_ind][course_ind][0]] * (
                        max_len_rf - len(rf_leaf_test[tree_ind][course_ind])) + list(
                    rf_leaf_test[tree_ind][course_ind])
    test_course_input_rf = []
    for t_i, t in zip(range(len(rf_leaf_test)), rf_leaf_test):
        for c_i, course in zip(range(len(t)), t):
            l = list(map(lambda x: w2v_model_rf.wv[x], [str(x) for x in course]))
            test_course_input_rf.append(l)

    # xgb_leaf_test = get_instances_route(trained_model_xgb, test_x, not is_gbt, tree_dic_xgb)[0]
    # for tree_ind, tree in zip(range(len(xgb_leaf_test)), xgb_leaf_test):
    #     for course_ind, course in zip(range(len(tree)), tree):
    #         if len(xgb_leaf_test[tree_ind][course_ind]) < max_len_xgb:
    #             xgb_leaf_test[tree_ind][course_ind] = [xgb_leaf_test[tree_ind][course_ind][0]] * (
    #                     max_len_xgb - len(xgb_leaf_test[tree_ind][course_ind])) + list(
    #                 xgb_leaf_test[tree_ind][course_ind])
    # test_course_input_xgb = []
    # for t_i, t in zip(range(len(xgb_leaf_test)), xgb_leaf_test):
    #     for c_i, course in zip(range(len(t)), t):
    #         l = list(map(lambda x: w2v_model_xgb.wv[x], [str(x) for x in course]))
    #         test_course_input_xgb.append(l)

    test_course_input_rf = np.array(test_course_input_rf)
    # test_course_input_xgb = np.array(test_course_input_xgb)
    test_input_net = pd.concat([test_input_net] * rf_estimators, ignore_index=True)
    test_y_net = pd.concat([targets_test] * rf_estimators, ignore_index=True)

    # loss, main_loss, aux_loss, sec_aux_loss, main_acc, aux_acc, sec_aux_acc = model_deep_forest.evaluate(
    #     [test_course_input_rf, test_input_net, test_course_input_xgb], [test_y_net, test_y_net, test_y_net])
    if not IS_MULTICLASS:
        test_probs = model_deep_forest.predict([test_course_input_rf, test_input_net])
        test_prob_list = list(test_probs[0])
        final_preds_average = []
        for i in range(test_x.shape[0]):
            preds_i = test_prob_list[i::test_x.shape[0]]
            final_preds_average.append(np.average(preds_i))
        # RF Embedding
        preds_e = []
        for li in test_probs[0]:
            proba = float(li[0])
            if proba > 0.5:
                cl = 1
            else:
                cl = 0
            preds_e.append(cl)
        # print("Embeding rf:\n")
        # print(confusion_matrix(test_y_net, preds_e))
        # XGB Embedding
        # preds_e = []
        # for li in test_probs[2]:
        #     proba = float(li[0])
        #     if proba > 0.5:
        #         cl = 1
        #     else:
        #         cl = 0
        #     preds_e.append(cl)
        # # print("Embeding xgb:\n")
        # print(confusion_matrix(test_y_net, preds_e))
        # Final prediction
        preds = []
        for li in test_probs[1]:
            proba = float(li[0])
            if proba > 0.5:
                cl = 1
            else:
                cl = 0
            preds.append(cl)
        # print("\nFinal prediction:\n")
        # print(confusion_matrix(test_y_net, preds))

        # print("**********************")
        # print('Forest encoding test accuracy:', accuracy_score(test_y_net, preds))
        # print("**********************")
        # print("\nAUC:\n")
        fe_acu = roc_auc_score(test_y, final_preds_average)
        # print(fe_acu)

        # Check rivals performnces
        xgb_rival_mx_depth = 10
        xgb_rival_n_est = 2000
        rf_rival_mx_depth = 10
        rf_rival_n_est = 2000

        start_rf = time()
        rf_rival_model = random_forest_classifier(auxiliary_train_input_data_for_rivals, train_y, rf_rival_n_est,
                                                  rf_rival_mx_depth)
        elapse_rf = time() - start_rf
        preds_rf = rf_rival_model.predict_proba(test_input_rivals)
        auc_rf = roc_auc_score(test_y, preds_rf[:, 1])

        start_xgb = time()
        xgb_model_rival = xgboost_classifier(auxiliary_train_input_data_for_rivals, train_y, xgb_rival_n_est,
                                             xgb_rival_mx_depth)
        elapse_xgb = time() - start_xgb
        preds_xgb = xgb_model_rival.predict_proba(test_input_rivals)
        auc_xgb = roc_auc_score(test_y, preds_xgb[:, 1])

        start_nn = time()
        nn_rival_model = simple_nn_classifier(auxiliary_train_input_data_for_rivals, train_y)
        elapse_nn = time() - start_nn
        preds_nn = nn_rival_model.predict_proba(test_input_rivals)
        auc_nn = roc_auc_score(test_y, preds_nn)
        del dataset
        del train_x
        del test_x
        K.clear_session()
        return auc_rf, elapse_rf, \
               auc_xgb, elapse_xgb, \
               auc_nn, elapse_nn, \
               fe_acu, fe_elapse, embedding_layer_size
    else:
        preds = []
        for x in model_deep_forest.predict(
                [get_instances_route(trained_model, test_x, is_gbt, tree_dic_rf)[0], test_input_net])[1]:
            preds.append(list(x).index(max(x)))
        macro_roc_auc_ovo = roc_auc_score(test_y, preds,
                                          average="macro")
        weighted_roc_auc_ovo = roc_auc_score(test_y, preds,
                                             average="weighted")
        print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
              "(weighted by prevalence)"
              .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
        return accuracy_score(test_y, trained_model.predict(
            test_x)), accuracy_score(test_y, preds), model_deep_forest, \
               trained_model, embedding_layer_size, 0, macro_roc_auc_ovo


def print_rivals_performances(rival_auc, rival_auc_xgb, test_x, test_y, train_x, train_y, trained_model,
                              trained_model_xgb):
    print("Train Accuracy rf :: " + str(accuracy_score(train_y, trained_model.predict(train_x))))
    print("Test Accuracy rf  :: " + str(accuracy_score(test_y, trained_model.predict(test_x))))
    print("Train Accuracy xgb :: " + str(accuracy_score(train_y, trained_model_xgb.predict(train_x))))
    print("Test Accuracy xgb  :: " + str(accuracy_score(test_y, trained_model_xgb.predict(test_x))))
    print("Test AUC rf:: " + str(rival_auc))
    print("Confusion matrix rf \n" + str(confusion_matrix(test_y, trained_model.predict(test_x))))
    print("Test AUC xg:: " + str(rival_auc_xgb))
    print("Confusion matrix xgb \n" + str(confusion_matrix(test_y, trained_model_xgb.predict(test_x))))


def print_dataset_details(test_x, test_y, train_x, train_y):
    print("Train_x Shape :: " + str(train_x.shape))
    print("Train_y Shape :: " + str(train_y.shape))
    print("Test_x Shape :: " + str(test_x.shape))
    print("Test_y Shape :: " + str(test_y.shape))
    print(len(train_y.value_counts()))
    print(len(test_y.value_counts()))


summary_results = []
win_rf = []
win_xgb = []
win_nn = []
data_folder = r"dataConv"
onlyfilenames = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
start_index = 0
for file_n in onlyfilenames[start_index:]:
    print("************************")
    print(str(start_index) + " of " + str(len(onlyfilenames)) + " done")
    print(file_n)
    print("************************")
    start_index += 1
    INPUT_PATH = join(data_folder, file_n)
    dataset_name = file_n
    rf_mses = []
    xgb_mses = []
    nn_mses = []
    fe_mses = []

    rf_elapses = []
    xgb_elapses = []
    nn_elapses = []
    fe_elapses = []

    total_time = time()
    num_of_folds = 3
    for iteration in range(num_of_folds):
        # print("The current fold is: " + str(iteration))
        rf_mse, rf_elapse, xgb_mse, xgb_elapse, nn_mse, nn_elapse, fe_mse, fe_elapse, embedding_size = run(iteration,
                                                                                                           INPUT_PATH)

        rf_mses.append(rf_mse)
        xgb_mses.append(xgb_mse)
        nn_mses.append(nn_mse)
        fe_mses.append(fe_mse)

        rf_elapses.append(rf_elapse)
        xgb_elapses.append(xgb_elapse)
        nn_elapses.append(nn_elapse)
        fe_elapses.append(fe_elapse)

    print("Dataset name: " + str(dataset_name))
    print("Number of folds: " + str(num_of_folds))
    print("Number of Epocs: " + str(NUM_OF_EPOCS))
    print("Number of Estimators: " + str(xgb_estimators))
    print("Max depth: " + str(xgb_depth))
    print("Embeding layer size: " + str(embedding_size))

    print("RF Test average AUC: {}".format(sum(rf_mses) / len(rf_mses)))
    print("RF average train time {}".format(sum(rf_elapses) / len(rf_elapses)))

    print("XGB Test average AUC: {}".format(sum(xgb_mses) / len(xgb_mses)))
    print("XGB average train time {}".format(sum(xgb_elapses) / len(xgb_elapses)))

    print("NN Test average AUC: {}".format(sum(nn_mses) / len(nn_mses)))
    print("NN average train time {}".format(sum(nn_elapses) / len(nn_elapses)))

    print("FE Test average AUC: {}".format(sum(fe_mses) / len(fe_mses)))
    print("FE average train time {}".format(sum(fe_elapses) / len(fe_elapses)))

    win_rf.append(sum(fe_mses) / len(fe_mses) >= sum(rf_mses) / len(rf_mses))
    win_xgb.append(sum(fe_mses) / len(fe_mses) >= sum(xgb_mses) / len(xgb_mses))
    win_nn.append(sum(fe_mses) / len(fe_mses) >= sum(nn_mses) / len(nn_mses))
    print("====Sumarry====")
    print("win_rf: " + str(sum(win_rf) / len(win_rf)))
    print("win_xgb: " + str(sum(win_xgb) / len(win_xgb)))
    print("win_nn: " + str(sum(win_nn) / len(win_nn)))
    print("===============")

    results = pd.DataFrame.from_records(
        [rf_mses, rf_elapses, xgb_mses, xgb_elapses, nn_mses, nn_elapses, fe_mses, fe_elapses])
    res_file = open(Path("Results/" + str(file_n) + ".csv"), 'w')
    results.to_csv(res_file)
    res_file.close()
    summary_results = [[file_n, sum(rf_mses) / len(rf_mses), sum(rf_elapses) / len(rf_elapses),
                        sum(xgb_mses) / len(xgb_mses), sum(xgb_elapses) / len(xgb_elapses),
                        sum(nn_mses) / len(nn_mses), sum(nn_elapses) / len(nn_elapses),
                        sum(fe_mses) / len(fe_mses), sum(fe_elapses) / len(fe_elapses)]]
    final_res = open(Path("Results/summary.csv"), 'a')
    df = pd.DataFrame(summary_results).to_csv(final_res, mode='a', header=False)
    final_res.close()
    del df
    del results
