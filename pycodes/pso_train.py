import pyswarms as ps
import pso_utility as pu
import numpy as np
import pandas as pd
import pickle
import configs as cfg

import matplotlib.pyplot as plt
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

def get_pso_weights():

    num_classifs = cfg.params_dict["num_classifs"]
    num_classes = cfg.params_dict["num_classes"]

    print("Optimizing...")

    # Initialize swarm
    options = {'c1': 1.419, 'c2': 1.419, 'w': 0.9}

    # Call instance of PSO
    dimensions = num_classes * num_classifs

    max_bound = np.ones(dimensions)
    min_bound = np.zeros(dimensions)
    bounds = (min_bound, max_bound)

    # print(bounds)

    loaded_arr = pickle.load(open("../models/" + cfg.params_dict["dataset"] + "/prediction_matrix_ml_without_xg_svm_with_ulmfit.sav", 'rb'))
    print("loaded_arr.shape", loaded_arr.shape)

    # colnames = ["cat", "useless", "content"]
    # Read Train Data
    df_train = pd.read_csv("../data/" + cfg.params_dict["dataset"] + "/train.csv", names=cfg.params_dict["colnames"], header=None)
    
    # Lables for Train data
    labels_train = np.array(list(df_train["cat"]))

    labels_train = labels_train - 1

    print("labels_train.shape", labels_train.shape)

    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options, bounds=bounds)

    kwargs={"kfold_res": loaded_arr, 'y_true':labels_train}

    # Perform optimization
    cost, pos = optimizer.optimize(pu.fa, iters=100, **kwargs)

    
    filename = "optimized_weights_ml_without_xg_svm_with_ulmfit.sav"
    pickle.dump(pos, open("../features/" + cfg.params_dict["dataset"] + "/" + filename, 'wb'))

    # plot_cost_history(cost_history=optimizer.cost_history)
    # # plt.show()
    # plt.savefig("../results/cost_history_" + cfg.params_dict["dataset"] + ".png", dpi=300)

    return pos

# get_pso_weights()