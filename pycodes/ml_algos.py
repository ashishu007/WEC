from sklearn import ensemble
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from scipy.sparse import csr_matrix
import xgboost
import pickle
from sklearn.metrics import accuracy_score, f1_score

import configs as cfg


def predict_classif(X, y, classif_name, feature_name):

    print(classif_name + " " + feature_name + " Prediction")
    loaded_model = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/" + classif_name + "_" + feature_name + ".sav", 'rb'))
    pred = loaded_model.predict(X)
    print("Accuracy", (pred == y).mean())
    print("Macro F1", f1_score(pred, y, average="macro"))

    # Predict the probabilities of each class
    pred_proba = loaded_model.predict_proba(X)

    return pred_proba

# Random Forest
def train_rf(x_train, y_train, ft):

    print("Random Forest " + ft + " Training")
    rf = ensemble.RandomForestClassifier(n_estimators=100)
    rf_model = rf.fit(x_train, y_train)
    
    # Saving the Random Forest Model trained in each fold
    filename = 'rf_' + ft + '.sav'
    pickle.dump(rf_model, open("../models/" + cfg.params_dict["dataset"] + "/" + filename, 'wb'))

    return "RF Model Trained and Saved"

# Naive Bayes
def train_nb(x_train, y_train, ft):

    print("Naive Bayes " + ft + " Training")
    nb = MultinomialNB()
    nb_model = nb.fit(x_train, y_train)
    
    # Saving the Naive Bayes Model trained in each fold
    filename = 'nb_' + ft + '.sav'
    pickle.dump(nb_model, open("../models/" + cfg.params_dict["dataset"] + "/" + filename, 'wb'))

    return "NB Model Trained and Saved"


# Logistic Regression
def train_lr(x_train, y_train, ft):

    print("Logisitic Regression " + ft + " Training")
    lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    lr_model = lr.fit(x_train, y_train)
  
    # Saving the Logistic Regression Model trained in each fold
    filename = 'lr_' + ft + '.sav'
    pickle.dump(lr_model, open("../models/" + cfg.params_dict["dataset"] + "/" + filename, 'wb'))

    return "LR Model Trained and Saved"


# SVM Linear
def train_svm(x_train, y_train, ft):
    
    print("SVM " + ft + " Training")
    svml = svm.SVC(kernel='linear', probability=True)
    svml_model = svml.fit(x_train, y_train)

    # Saving the XgBoost Model trained in each fold
    filename = 'svm_' + ft + '.sav'
    pickle.dump(svml_model, open("../models/" + cfg.params_dict["dataset"] + "/" + filename, 'wb'))

    return "SVM Model Trained and Saved"


# Xg Boost
def train_xgb(x_train, y_train, ft):

    print("XgBoost " + ft + " Training")
    xg = xgboost.XGBClassifier()
    xg_model = xg.fit(csr_matrix(x_train), y_train)

    # Saving the XgBoost Model trained in each fold
    filename = 'xg_' + ft + '.sav'
    pickle.dump(xg_model, open("../models/" + cfg.params_dict["dataset"] + "/" + filename, 'wb'))

    return "XgB Model Trained and Saved"
