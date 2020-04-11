import pandas as pd
import numpy as np
import generate_features as gf
import generate_embs as ge
import kfold as kf
from keras.utils import to_categorical
import ml_algos as ml
import dl_algos as dl
import configs as cfg


def train_whole(tf_X, tfidf_X, we_X, y, word_index_len):
# def train_whole(tf_X, tfidf_X, y):

    # Target value for dl algos in One-Hot Encoding format
    y_train_cat = to_categorical(np.asarray(y))

    # # Define the train and validation data on best top features TF
    # train_x_tf = np.array(pd.DataFrame(tf_X))

    # # Define the train and validation data on best top features TFIDF
    # train_x_tfidf = np.array(pd.DataFrame(tfidf_X))

    # # Training different ML Algos on TF
    # ml.train_rf(train_x_tf, y, "TF")
    # ml.train_nb(train_x_tf, y, "TF")
    # ml.train_xgb(train_x_tf, y, "TF")
    # ml.train_svm(train_x_tf, y, "TF")
    # ml.train_lr(train_x_tf, y, "TF")

    # # Training different ML Algos on TFIDF
    # ml.train_rf(train_x_tfidf, y, "TFIDF")
    # ml.train_nb(train_x_tfidf, y, "TFIDF")
    # ml.train_xgb(train_x_tfidf, y, "TFIDF")
    # ml.train_svm(train_x_tfidf, y, "TFIDF")
    # ml.train_lr(train_x_tfidf, y, "TFIDF")

    weight_matrix = ge.generate_glove_embeddings(word_index_len)

    # # Training DL Algos on random WE
    cnn = dl.train_cnn(we_X, y_train_cat, word_index_len, weight_matrix, "T")
    # blstm = dl.train_blstm(we_X, y_train_cat, "None", "None", word_index_len, 4, weight_matrix, "T")
    # reccnn = dl.train_reccnn(we_X, y_train_cat, "None", "None", word_index_len, 4, weight_matrix, "T")

    # # Training DL Algos on glove WE
    cnn_glove = dl.train_cnn(we_X, y_train_cat, word_index_len, weight_matrix, "F")
    # blstm_glove = dl.train_blstm(we_X, y_train_cat, "None", "None", word_index_len, 4, weight_matrix, "F")
    # reccnn_glove = dl.train_reccnn(we_X, y_train_cat, "None", "None", word_index_len, 4, weight_matrix, "F")

    return "Saved for whole train data"