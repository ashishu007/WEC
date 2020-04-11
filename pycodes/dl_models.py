from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D
from keras.layers import MaxPool1D, Concatenate, Flatten, Dropout, Dense, Conv1D
from keras.layers import LSTM, Bidirectional, Embedding
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, Adadelta
from keras import regularizers

import numpy as np
import pandas as pd
import configs as cfg

# Defining the CNN model from paper 
# Convolutional Neural Networks for Sentence Classification
def get_cnn(word_index_len, embedding_matrix, training):

    num_classes = cfg.params_dict["num_classes"]
    num_filters = 108
    filter_sizes = [3, 4, 5]
    embedding_dim = cfg.params_dict["embedding_dim"]
    MAX_SEQUENCE_LENGTH = cfg.params_dict["max_seq_length"]

    if training == "F":
        embedding_layer = Embedding(word_index_len + 1,
                                    embedding_dim,
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    weights=[embedding_matrix], trainable=False)

    if training == "T":
        embedding_layer = Embedding(word_index_len + 1,
                                    embedding_dim,
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    weights=[embedding_matrix], trainable=True)

    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding = embedding_layer(inputs)

    # print(embedding.shape)
    reshape = Reshape((MAX_SEQUENCE_LENGTH,embedding_dim,1))(embedding)
    # print(reshape.shape)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', 
                                                kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', 
                                                kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', 
                                                kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1), strides=(1,1), 
                                                padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1), strides=(1,1), 
                                                padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1), strides=(1,1), 
                                                padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(0.5)(flatten)
    out1 = Dense(units=1024, activation='sigmoid')(dropout)

    output = Dense(units=num_classes, activation='softmax')(out1)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    # adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adadelta = Adadelta(lr=0.95)

    model.compile(optimizer=adadelta, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


# Defining the Recurrent CNN Model from paper 
# Recurrent Convolutional Neural Networks for Text Classification
def RecCNN(word_index_len, embedding_matrix, training):

    num_classes = cfg.params_dict["num_classes"]
    embedding_dim = cfg.params_dict["embedding_dim"]
    MAX_SEQUENCE_LENGTH = cfg.params_dict["max_seq_length"]

    if training == "F":
        embedding_layer = Embedding(word_index_len + 1,
                                    embedding_dim,
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    weights=[embedding_matrix], trainable=False)

    if training == "T":
        embedding_layer = Embedding(word_index_len + 1,
                                    embedding_dim,
                                    input_length=MAX_SEQUENCE_LENGTH)

    inputs = Input(shape=[MAX_SEQUENCE_LENGTH], dtype='int32')
    embedding = embedding_layer(inputs)
    
    x = Dropout(0.25)(embedding)
    x = Conv1D(64,
             5,
             padding='valid',
             activation='relu',
             strides=1)(x)
    x = MaxPool1D(pool_size=4)(x)
    x = LSTM(70)(x)
    x = Dense(num_classes)(x)
    preds = Activation('softmax')(x)
    model = Model(inputs, preds)
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    
    return model


# Defin the BiLSTM-2DCNN model from paper 
# Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling
def blstm_2dcnn(word_index_len, embedding_matrix, training):

    num_classes = cfg.params_dict["num_classes"]
    embedding_dim = cfg.params_dict["embedding_dim"]
    MAX_SEQUENCE_LENGTH = cfg.params_dict["max_seq_length"]

    if training == "F":
        embedding_layer = Embedding(word_index_len + 1,
                                    embedding_dim,
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    weights=[embedding_matrix], trainable=False)

    if training == "T":
        embedding_layer = Embedding(word_index_len + 1,
                                    embedding_dim,
                                    input_length=MAX_SEQUENCE_LENGTH)
                            
    inputs = Input(shape=[MAX_SEQUENCE_LENGTH], dtype='int32')
    embedding = embedding_layer(inputs)
    x = Dropout(0.5)(embedding)
    x = Bidirectional(LSTM(
                        300,
                        # recurrent_dropout=0.2,
                        kernel_regularizer=regularizers.l2(1e-5),
                        return_sequences=True))(x)
    
    x = Dropout(0.2)(x)
    x = Reshape((2 * MAX_SEQUENCE_LENGTH, 300, 1))(x)
    x = Conv2D(100, (3, 3), kernel_regularizer=regularizers.l2(1e-5))(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, preds)
    adadelta = Adadelta(lr=1.0)
    model.compile(loss='categorical_crossentropy',optimizer=adadelta,metrics=['accuracy'])
    model.summary()
    
    return model