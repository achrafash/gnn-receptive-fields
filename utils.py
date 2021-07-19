import os
import pandas as pd
import torch
import torch.nn as nn
import yaml


def save_experiment(path: str, model: torch.nn.Module, outs: pd.DataFrame, config: dict, comment: str):
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
    with open(path + "config.yml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    outs.to_csv(path + "logs.csv", index=False)

    with open(path + "comment.txt", "w") as f:
        f.write(comment)
        
    print("Done")
    return


def load_experiment(path:str):
    # load model
    model = torch.load(path+'model.pt')
    # load configs
    with open(path + "config.yml", "r") as stream:
        config = yaml.full_load(stream)
    # load logs
    logs = pd.read_csv(path+'logs.csv')

    return model, config, logs
