import yaml
import pandas as pd
import json

def load_params():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    return params


def save_params(params):
    with open("params.yaml", "w") as f:
        yaml.dump(params, f, sort_keys=False)


def load_data(path):
    return pd.read_csv(path)


def save_data(data, path):
    data.to_csv(path)


def save_json(dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict, f, indent=2)