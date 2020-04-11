import pandas as pd

dataset_name = "newsgroup"
colnames = ["cat", "content"]
df = pd.read_csv("../data/" + dataset_name + "/train.csv", names=colnames)
num_classes = len(set(list(df["cat"])))

params_dict = {
    "dataset": dataset_name,
    "colnames": colnames,
    "num_classifs": 7,
    "num_classes": num_classes,
    "max_seq_length": 50,
    "embedding_dim": 300
}