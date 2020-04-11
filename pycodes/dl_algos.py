import dl_models as dlm
import numpy as np
import configs as cfg
from keras.models import load_model


def predict_classif_dl(X, y, classif_name, pre_trained):
    
    model_name = classif_name + "_" + pre_trained + ".h5"
    loaded_model = load_model("../models/" + cfg.params_dict["dataset"] + "/" + model_name)

    pred_proba = loaded_model.predict(X)

    pred_y = np.argmax(pred_proba, axis=1)
    print("Accuracy ", model_name, (pred_y == y).mean())

    return pred_proba


def train_cnn(x_train, y_train, word_index_len, weight_matrix, training):

    # Call the model
    cnn_model = dlm.get_cnn(word_index_len, weight_matrix, training)
    
    # Train the model
    print("CNN Training") 
    cnn_model.fit(x_train, y_train, validation_split=0.33, batch_size=32, epochs=10, verbose=1)

    # Save the model
    filename = 'cnn' + "_" + training + '.h5'
    cnn_model.save("../models/" + cfg.params_dict["dataset"] + "/" + filename)

    return "CNN Model Saved"


def train_blstm(x_train, y_train, word_index_len, weight_matrix, training):

    # Call the model
    blstm_model = dlm.blstm_2dcnn(word_index_len, weight_matrix, training)
    
    # Train the model
    print("BLSTM Training") 
    blstm_model.fit(x_train, y_train, validation_split=0.33, batch_size=32, epochs=25, verbose=1)

    # Save the model
    filename = 'blstm' + training + '.h5'
    blstm_model.save("../models/" + cfg.params_dict["dataset"] + "/" + filename)

    return "BLSTM Model Saved"


def train_reccnn(x_train, y_train, word_index_len, weight_matrix, training):

    # Call the model
    rec_model = dlm.RecCNN(word_index_len, weight_matrix, training)
    
    # Train the model
    print("RecCNN Training") 
    rec_model.fit(x_train, y_train, validation_split=0.33, batch_size=32, epochs=25, verbose=1)

    # Save the model
    filename = 'RecCNN' + training + '.h5'
    rec_model.save("../models/" + cfg.params_dict["dataset"] + "/" + filename)

    return "RecCNN Model Saved"