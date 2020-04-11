from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import pickle
import configs as cfg

def train_feature_vector(X, ft):
    if ft == "TFIDF":
        vect = TfidfVectorizer(analyzer='word', stop_words='english', max_features=5000)
    if ft == "TF":
        vect = CountVectorizer(analyzer='word', stop_words='english', max_features=5000)
    vect.fit(X)

    pickle.dump(vect, open("../features/" + cfg.params_dict["dataset"] + "/" + ft + "_Features.sav", 'wb'))

def generate_feature_vector(X, ft):
    vect = pickle.load(open("../features/" + cfg.params_dict["dataset"] + "/" + ft + "_Features.sav", 'rb'))

    x_vect = vect.transform(X)

    x_df = pd.DataFrame(x_vect.toarray(), columns=vect.get_feature_names())

    print("x_df.shape", ft, x_df.shape)
    
    return x_df

def feature_selection(X, y, fold, ft):
    # Create and fit selector
    selector = SelectKBest(chi2, k=2000)
    selector.fit(X, y)
    mask = selector.get_support()
    new_features = X.columns[mask]

    filename = "chi_fs_" + ft + str(fold) + ".sav"
    pickle.dump(new_features, open("../features/" + cfg.params_dict["dataset"] + "/" + filename, 'wb'))
    
    return new_features
