import pickle 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import configs as cfg

# Tokenize the train set and create vocabulary of present words in the corpus
def tokenize_text(X):
    tokenizer  = Tokenizer()
    tokenizer.fit_on_texts(X)

    pickle.dump(tokenizer, open("../features/" + cfg.params_dict["dataset"] + "/Tokenizer.sav", 'wb'))

    return len(tokenizer.word_index)


def generate_random_embeddings(X, seq_len):

    tokenizer = pickle.load(open("../features/" + cfg.params_dict["dataset"] + "/Tokenizer.sav", 'rb'))

    tokenizer.fit_on_texts(X)
    
    # Tokenize the train_x and test_x strings
    seq =  tokenizer.texts_to_sequences(X)

    # Pad the tokenized train_x and test_x strings into sequences of length 50
    pad_seq = pad_sequences(seq, maxlen=seq_len)

    return pad_seq


def generate_glove_embeddings(vocab_size):

    tokenizer = pickle.load(open("../features/" + cfg.params_dict["dataset"] + "/Tokenizer.sav", 'rb'))

    print(type(tokenizer))

    embeddings_index = dict()
    f = open('../datasets/glove.6B.300d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size + 1, cfg.params_dict["embedding_dim"])) # +1 is to avoid going out of index
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix
