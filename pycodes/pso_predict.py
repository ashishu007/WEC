import pickle, json
import pandas as pd
import numpy as np
import pso_utility as pu
import pso_train as pt
from keras.models import load_model
import generate_embs as ge
from scipy.sparse import csr_matrix
from sklearn import metrics
import configs as cfg

pt.get_pso_weights()

def predict_final_res():

    # f = open("../misc/filename.txt", "r")
    # temp = f.read()
    # pos = np.array(temp)

    # x = []
    # file_in = open('../misc/filename.txt', 'r')

    # for y in file_in.read().split(','):
    #     x.append(float(y))

    # pos = np.array(x)
    # # print(x.shape)


    pos = pickle.load(open("../features/" + cfg.params_dict["dataset"] + "/optimized_weights_ml_without_xg_svm_with_ulmfit.sav", 'rb'))
    print(pos.shape)

    # Read Test Data
    df_test = pd.read_csv("../data/" + cfg.params_dict["dataset"] + "/test.csv", names=cfg.params_dict["colnames"], header=None)
    # Lables for Test data
    labels_test = np.array(list(df_test["cat"]))

    # labels_test = labels_test - 1

    print("Test Labels")
    print(set(labels_test))

    # Load the TF and TFIDF Vectorizers
    tf = pickle.load(open("../features/" + cfg.params_dict["dataset"] + "/TF_Features.sav", 'rb'))
    tfidf = pickle.load(open("../features/" + cfg.params_dict["dataset"] + "/TFIDF_Features.sav", 'rb'))

    # Transform Test set into TF and TFIDF Vectors
    tf_vect = tf.transform(df_test["content"])
    tfidf_vect = tfidf.transform(df_test["content"])

    test_x_tf = np.array(pd.DataFrame(tf_vect.toarray(), columns=tf.get_feature_names()))
    test_x_tfidf = np.array(pd.DataFrame(tfidf_vect.toarray(), columns=tfidf.get_feature_names()))
    # static_emb_test = ge.generate_random_embeddings(df_test["content"], cfg.params_dict["max_seq_length"])


    # Predicting probabilities for test set from the models trained on whole data
    loaded_nb_tf = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/" + "nb_TF.sav", 'rb'))
    loaded_rf_tf = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/" + "rf_TF.sav", 'rb'))
    loaded_lr_tf = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/" + "lr_TF.sav", 'rb'))
    # loaded_svm_tf = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/" + "svm_TF.sav", 'rb'))
    # loaded_xg_tf = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/" + "xg_TF.sav", 'rb'))

    loaded_nb_tfidf = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/" + "nb_TFIDF.sav", 'rb'))
    loaded_rf_tfidf = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/" + "rf_TFIDF.sav", 'rb'))
    loaded_lr_tfidf = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/" + "lr_TFIDF.sav", 'rb'))
    # loaded_svm_tfidf = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/" + "svm_TFIDF.sav", 'rb'))
    # loaded_xg_tfidf = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/" + "xg_TFIDF.sav", 'rb'))

    # print(test_x_tf)
    # print(test_x_tfidf)
    # print(static_emb_test)

    # loaded_cnn = load_model("../models/" + cfg.params_dict["dataset"] + "/" + 'cnn_T.h5')
    # loaded_blstm = load_model("../models/" + 'blstmT.h5')
    # loaded_reccnn = load_model("../models/" + 'RecCNNT.h5')

    # loaded_cnn_p = load_model("../models/" + cfg.params_dict["dataset"] + "/" + 'cnn_F.h5')
    # loaded_blstm_p = load_model("../models/" + 'blstmF.h5')
    # loaded_reccnn_p = load_model("../models/" + 'RecCNNF.h5')

    print("Predicting Probability ...")

    r_tf = loaded_rf_tf.predict_proba(test_x_tf)
    n_tf = loaded_nb_tf.predict_proba(test_x_tf)
    l_tf = loaded_lr_tf.predict_proba(test_x_tf)
    # x_tf = loaded_xg_tf.predict_proba(csr_matrix(test_x_tf))
    # x_tfidf = loaded_xg_tfidf.predict_proba(csr_matrix(test_x_tfidf))

    # s_tf = loaded_svm_tf.predict_proba(test_x_tf)

    r_tfidf = loaded_rf_tfidf.predict_proba(test_x_tfidf)
    n_tfidf = loaded_nb_tfidf.predict_proba(test_x_tfidf)
    l_tfidf = loaded_lr_tfidf.predict_proba(test_x_tfidf)
    # s_tfidf = loaded_svm_tfidf.predict_proba(test_x_tfidf)

    # c_pred = loaded_cnn.predict(static_emb_test)
    # blstm_pred = loaded_blstm.predict(static_emb_test)
    # reccnn_pred = loaded_reccnn.predict(static_emb_test)

    # c_pred_p = loaded_cnn_p.predict(static_emb_test)
    # blstm_pred_p = loaded_blstm_p.predict(static_emb_test)
    # reccnn_pred_p = loaded_reccnn_p.predict(static_emb_test)

    # Stacking the predictions from 14 Algos (num_classifiers) 
    # with 4 classes (num_classes)
    # into a matrix of (num_samples x num_classifiers x num_classes)

    # bert_pred = pickle.load(open("../misc/bert_prediction_on_test_" + cfg.params_dict["dataset"] + ".sav", 'rb'))
    # bert_pred = np.exp(bert_pred)

    ulmfit_pred = pickle.load(open("../misc/ulmfit_prediction_on_test_" + cfg.params_dict["dataset"] + ".sav", 'rb'))

    prediction_matrix_fold = np.stack((r_tf, 
                                    n_tf, 
                                    # s_tf, 
                                    l_tf, 
                                    r_tfidf, 
                                    n_tfidf, 
                                    # s_tfidf, 
                                    l_tfidf, 
                                    # bert_pred,
                                    ulmfit_pred#,
                                    # x_tf, 
                                    # x_tfidf, 
                                    # c_pred, #blstm_pred, reccnn_pred,
                                    # c_pred_p#, blstm_pred_p, reccnn_pred_p
                                    ), 
                                    axis=1)

    print(prediction_matrix_fold.shape)


    acc_dict = {}
    mf1_dict = {}

    # Prediction for PSO
    # print("PSO Prediction")
    acc, f1 = pu.predict(pos, prediction_matrix_fold, labels_test)
    # print("Acc", acc)
    # print("F1", f1)
    acc_dict["PSO"] = acc
    mf1_dict["PSO"] = f1

    print(acc, f1)

    # labels_test = labels_test + 1

    # # print("Random Forest Prediction")
    # rf_pred = loaded_rf_tf.predict(test_x_tf)    
    # acc_dict["RF TF"] = (rf_pred == labels_test).mean()
    # mf1_dict["RF TF"] = metrics.f1_score(rf_pred, labels_test, average="macro")

    # # print("Naive Bayes Prediction")
    # nb_pred = loaded_nb_tfidf.predict(test_x_tfidf)    
    # acc_dict["NB TFIDF"] = (nb_pred == labels_test).mean()
    # mf1_dict["NB TFIDF"] = metrics.f1_score(nb_pred, labels_test, average="macro")

    # # print("Logistic Regression Prediction")
    # lr_pred = loaded_lr_tf.predict(test_x_tf)    
    # acc_dict["LR TF"] = (lr_pred == labels_test).mean()
    # mf1_dict["LR TF"] = metrics.f1_score(lr_pred, labels_test, average="macro")

    # # # print("Xg Boost Prediction")
    # # xg_pred = loaded_xg_tf.predict(csr_matrix(test_x_tf))    
    # # acc_dict["Xg TF"] = ((xg_pred == labels_test).mean())
    # # mf1_dict["Xg TF"] = (metrics.f1_score(xg_pred, labels_test, average="macro"))

    # # # print("SVM Linear Prediction")
    # # svm_pred = loaded_svm_tf.predict(test_x_tf)
    # # acc_dict["SVM TF"] = ((svm_pred == labels_test).mean())
    # # mf1_dict["SVM TF"] = (metrics.f1_score(svm_pred, labels_test, average="macro"))

    # # print("Random Forest Prediction")
    # rf_pred = loaded_rf_tfidf.predict(test_x_tfidf)
    # acc_dict["RF TFIDF"] = ((rf_pred == labels_test).mean())
    # mf1_dict["RF TFIDF"] = (metrics.f1_score(rf_pred, labels_test, average="macro"))

    # # print("Naive Bayes Prediction")
    # nb_pred = loaded_nb_tf.predict(test_x_tf)
    # acc_dict["NB TF"] = ((nb_pred == labels_test).mean())
    # mf1_dict["NB TF"] = (metrics.f1_score(nb_pred, labels_test, average="macro"))

    # # print("Logistic Regression Prediction")
    # lr_pred = loaded_lr_tfidf.predict(test_x_tfidf)
    # acc_dict["LR TFIDF"] = ((lr_pred == labels_test).mean())
    # mf1_dict["LR TFIDF"] = (metrics.f1_score(lr_pred, labels_test, average="macro"))

    # # # print("Xg Boost Prediction")
    # # xg_pred = loaded_xg_tfidf.predict(csr_matrix(test_x_tfidf))
    # # acc_dict["Xg TFIDF"] = ((xg_pred == labels_test).mean())
    # # mf1_dict["Xg TFIDF"] = (metrics.f1_score(xg_pred, labels_test, average="macro"))

    # # # print("SVM Linear Prediction")
    # # svm_pred = loaded_svm_tfidf.predict(test_x_tfidf)
    # # acc_dict["SVM TFIDF"] = ((svm_pred == labels_test).mean())
    # # mf1_dict["SVM TFIDF"] = (metrics.f1_score(svm_pred, labels_test, average="macro"))

    # # # print("CNN Prediction")
    # # cnn_pred = loaded_cnn.predict(static_emb_test)
    # # cnn_pred_arg = np.argmax(cnn_pred, axis=1)
    # # acc_dict["CNN"] = ((cnn_pred_arg == labels_test).mean())
    # # mf1_dict["CNN"] = (metrics.f1_score(cnn_pred_arg, labels_test, average="macro"))

    # # cnn_pred = loaded_cnn.predict(static_emb_test)
    # # cnn_pred_arg = np.argmax(cnn_pred, axis=1)
    # # print((cnn_pred_arg == labels_test).mean())

    # # print(cnn_pred)
    # # print(cnn_pred_arg)

    # # # print("CNN Pre Trained Prediction")
    # # cnn_pred_p = loaded_cnn_p.predict(static_emb_test)
    # # cnn_pred_arg_p = np.argmax(cnn_pred_p, axis=1)
    # # acc_dict["CNN Pre"] = ((cnn_pred_arg_p == labels_test).mean())
    # # mf1_dict["CNN Pre"] = (metrics.f1_score(cnn_pred_arg_p, labels_test, average="macro"))

    # # cnn_pred_p = loaded_cnn_p.predict(static_emb_test)
    # # cnn_pred_arg_p = np.argmax(cnn_pred_p, axis=1)
    # # print((cnn_pred_arg_p == labels_test).mean())

    # # print(acc_dict)
    # # print(mf1_dict)

    # # filename = "../results/" + cfg.params_dict["dataset"] + "_ml_without_xg_svm.json"

    # # with open(filename, 'w') as fout:
    # #     json.dump([acc_dict, mf1_dict], fout)

    # print(acc_dict)
    # print(mf1_dict)

    # return acc_dict, mf1_dict
    return acc, f1


a, m = predict_final_res()
# print(a, m)