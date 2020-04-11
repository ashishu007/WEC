import pandas as pd
import numpy as np
import os

import generate_features as gf
import generate_embs as ge
import kfold as kf
import pso_train as pt
import train_whole_data as twd
import configs as cfg

print("Libs Done")

data_dir = "../data/" + cfg.params_dict["dataset"]
features_dir = "../features/" + cfg.params_dict["dataset"]
models_dir = "../models/" + cfg.params_dict["dataset"]

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(features_dir):
    os.makedirs(features_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Read Train Data
df_train = pd.read_csv("../data/" + cfg.params_dict["dataset"] + "/train.csv", names=cfg.params_dict["colnames"], header=None)
# Read Test Data
df_test = pd.read_csv("../data/" + cfg.params_dict["dataset"] + "/test.csv", names=cfg.params_dict["colnames"], header=None)
print("Data Loaded")

# Lables for Train and Test data
labels_train = np.array(list(df_train["cat"]))
labels_test = np.array(list(df_test["cat"]))

labels_train = labels_train - 1
labels_test = labels_test - 1

# breakpoint()

print("len(labels_train), len(labels_test)", len(labels_train), len(labels_test))
print("set(labels_test), set(labels_train)", set(labels_test), set(labels_train))

# Train TF and TFIDF vectorizer on Train data
gf.train_feature_vector(df_train["content"], "TF")
gf.train_feature_vector(df_train["content"], "TFIDF")

# Transform Train sentences into TF and TFIDF vectors
tf_train = gf.generate_feature_vector(df_train["content"], "TF")
tfidf_train = gf.generate_feature_vector(df_train["content"], "TFIDF")

print("TF and TFIDF Done")

# Get the size of vocabulary for word embedding and train the Keras Tokenizer
vocab_len = ge.tokenize_text(df_train["content"])

# Embed the Train and Test sentences randomly. Here max_sequence_length is 50
static_emb_train = ge.generate_random_embeddings(df_train["content"], cfg.params_dict["max_seq_length"])
static_emb_test = ge.generate_random_embeddings(df_test["content"], cfg.params_dict["max_seq_length"])

print("Word Embedding Done")

# Collect the (num_classifiers x num_classes) matrix for TF and TFIDF vectors
L_matrix = kf.run_kfold(tf_train, tfidf_train, static_emb_train, labels_train, vocab_len)
# L_matrix = kf.run_kfold(tf_train, tfidf_train, labels_train)

print("L_matrix.shape", L_matrix.shape)

# breakpoint()

# print(labels_train)

# Train models for whole data
msg = twd.train_whole(tf_train, tfidf_train, static_emb_train, labels_train, vocab_len)
# msg = twd.train_whole(tf_train, tfidf_train, labels_train)

print(msg)


# Train the optimized weight matrix for PSO
# pos = pt.get_pso_weights()

# print(pos.shape)
