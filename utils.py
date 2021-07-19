import os
import pandas as pd
import torch
import torch.nn as nn
import yaml


def save_experiment(path: str, model: nn.Module, outs: pd.DataFrame, config: dict, comment: str):
    """
    Args:
        - path: directory for the experiment
        - model: trained torch model
        - outs: data frame of the results (loss, accuracy, and other metrics)
        - config: dictionary of the hyperparameters
        - comment: what is special about this experiment (more is always better)
    """
    # create the folder
    os.mkdir(path)

    # save the model
    torch.save(model, path + "model.pt")

    # save the config
    with open(path + "data.yml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    outs.to_csv(path + "outs.csv")

    with open(path + "comment.txt", "w") as f:
        f.write(comment)

    print("Done")
