import generate_features as gf
import generate_embs as ge
import ml_algos as ml
import dl_algos as dl
import configs as cfg

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from keras.utils import to_categorical

def run_kfold(tf_X, tfidf_X, we_X, y, word_index_len):
# def run_kfold(tf_X, tfidf_X, y):

    # Start 10-fold Cross Validation on Train data
    kf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)
    
    ctr = 0

    res = []

    # print("tf_X.shape", tf_X.shape)

    # print("tfidf_X.shape", tfidf_X.shape)

    print("y.shape", y.shape)

    for train_index, val_index in kf.split(we_X, y):
    # for train_index, val_index in kf.split(tf_X, y):

        print("Fold:", ctr)
        print("No of validation samples =", len(val_index), "No of training samples =", len(train_index))
        ctr += 1

        # Define train and validation data for TF
        X_train_tf, X_val_tf = tf_X.iloc[train_index], tf_X.iloc[val_index]

        # Define train and validation data for TFIDF
        X_train_tfidf, X_val_tfidf = tfidf_X.iloc[train_index], tfidf_X.iloc[val_index]

        print("X_train_tf.shape, X_val_tf.shape", X_train_tf.shape, X_val_tf.shape)
        print("X_train_tfidf.shape, X_val_tfidf.shape", X_train_tfidf.shape, X_val_tfidf.shape)

        # Define train and validation data for Word Embeddings
        X_train_we, X_val_we = we_X[train_index], we_X[val_index] 

        # Target value for ml algos
        y_train, y_val = y[train_index], y[val_index]

        print("y_train.shape, y_val.shape", y_train.shape, y_val.shape)

        # Target value for dl algos in One-Hot Encoding format
        y_train_cat = to_categorical(np.asarray(y_train))
        y_val_cat = to_categorical(np.asarray(y_val))

        # Define the train and validation data on best top features TF
        train_x_tf = np.array(pd.DataFrame(X_train_tf))
        val_x_tf = np.array(pd.DataFrame(X_val_tf))

        # Define the train and validation data on best top features TFIDF
        train_x_tfidf = np.array(pd.DataFrame(X_train_tfidf))
        val_x_tfidf = np.array(pd.DataFrame(X_val_tfidf))

        print("train_x_tf.shape, val_x_tf.shape", train_x_tf.shape, val_x_tf.shape)
        print("train_x_tfidf.shape, val_x_tfidf.shape", train_x_tfidf.shape, val_x_tfidf.shape)


        # # Training different ML Algos on TF
        # ml.train_rf(train_x_tf, y_train, "TF")
        # rf_tf = ml.predict_classif(val_x_tf, y_val, "rf", "TF")
        # print("rf_tf.shape", rf_tf.shape)

        # ml.train_nb(train_x_tf, y_train, "TF")
        # nb_tf = ml.predict_classif(val_x_tf, y_val, "nb", "TF")
        # print("nb_tf.shape", nb_tf.shape)

        # ml.train_xgb(train_x_tf, y_train, "TF")
        # xgb_tf = ml.predict_classif(val_x_tf, y_val, "xg", "TF")
        # print("xgb_tf.shape", xgb_tf.shape)
        
        # ml.train_svm(train_x_tf, y_train, "TF")
        # svm_tf = ml.predict_classif(val_x_tf, y_val, "svm", "TF")
        # print("svm_tfidf.shape", svm_tf.shape)
        
        # ml.train_lr(train_x_tf, y_train, "TF")
        # lr_tf = ml.predict_classif(val_x_tf, y_val, "lr", "TF")
        # print("lr_tf.shape", lr_tf.shape)


        # # Training different ML Algos on TFIDF
        # ml.train_rf(train_x_tfidf, y_train, "TFIDF")
        # rf_tfidf = ml.predict_classif(val_x_tfidf, y_val, "rf", "TFIDF")
        # print("rf_tfidf.shape", rf_tfidf.shape)

        # ml.train_nb(train_x_tfidf, y_train, "TFIDF")
        # nb_tfidf = ml.predict_classif(val_x_tfidf, y_val, "nb", "TFIDF")
        # print("nb_tfidf.shape", nb_tfidf.shape)

        # ml.train_xgb(train_x_tfidf, y_train, "TFIDF")
        # xgb_tfidf = ml.predict_classif(val_x_tfidf, y_val, "xg", "TFIDF")
        # print("xgb_tfidf.shape", xgb_tfidf.shape)

        # ml.train_svm(train_x_tfidf, y_train, "TFIDF")
        # svm_tfidf = ml.predict_classif(val_x_tfidf, y_val, "svm", "TFIDF")
        # print("svm_tfidf.shape", svm_tfidf.shape)

        # ml.train_lr(train_x_tfidf, y_train, "TFIDF")
        # lr_tfidf = ml.predict_classif(val_x_tfidf, y_val, "lr", "TFIDF")
        # print("lr_tfidf.shape", lr_tfidf.shape)

        weight_matrix = ge.generate_glove_embeddings(word_index_len)

        # Training DL Algos on random WE
        dl.train_cnn(X_train_we, y_train_cat, word_index_len, weight_matrix, "T")
        cnn = dl.predict_classif_dl(X_val_we, y_val, "cnn", "T")

        # dl.train_blstm(X_train_we, y_train_cat, word_index_len, weight_matrix, "T")
        # blstm = dl.predict_classif_dl(X_val_we, y_val, "blstm", "T")

        # dl.train_reccnn(X_train_we, y_train_cat, word_index_len, weight_matrix, "T")
        # reccnn = dl.predict_classif_dl(X_val_we, y_val, "RecCNN", "T")

        # Training DL Algos on glove WE
        dl.train_cnn(X_train_we, y_train_cat, word_index_len, weight_matrix, "F")
        cnn_glove = dl.predict_classif_dl(X_val_we, y_val, "cnn", "F")

        # dl.train_blstm(X_train_we, y_train_cat, word_index_len, weight_matrix, "F")
        # blstm_glove = dl.predict_classif_dl(X_val_we, y_val, "blstm", "F")

        # dl.train_reccnn(X_train_we, y_train_cat, word_index_len, weight_matrix, "F")
        # reccnn_glove = dl.predict_classif_dl(X_val_we, y_val, "RecCNN", "F")

        # Stacking the predictions from 14 Algos (num_classifiers) 
        # with 6 classes (num_classes)
        # into a matrix of (num_samples x num_classifiers x num_classes)
        prediction_matrix_fold = np.stack((
                                        # rf_tf, nb_tf, svm_tf, lr_tf, rf_tfidf, 
                                        # nb_tfidf, svm_tfidf, lr_tfidf, xgb_tf, xgb_tfidf, 
                                        cnn, #blstm, reccnn,
                                        cnn_glove #, blstm_glove, reccnn_glove
                                        ), 
                                        axis=1)

        print("prediction_matrix_fold_cnn_only.shape", prediction_matrix_fold.shape)

        res.append(prediction_matrix_fold)

    # Vertically stack all the matrices generated from each fold into one final matrix
    prediction_matrix = np.vstack(res)
    print("prediction_matrix_cnn.shape", prediction_matrix.shape)

    filename = 'prediction_matrix_cnn.sav'
    pickle.dump(prediction_matrix, open("../models/" + cfg.params_dict["dataset"] + "/" + filename, 'wb'))

    return prediction_matrix
    # return "hello"