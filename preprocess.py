import argparse
import csv
import matplotlib.pyplot as plt

import numpy as np
import pickle
import string

from box import Box

from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-c', '--config', dest='config_path',
            default='./config.yaml', type=Path,
            help='config file path')
    args = parser.parse_args()
    return vars(args)


def read_data(data_dir, dataset):
    data_file = Path(data_dir) / f"{dataset}.tsv"
    data = []
    with open(data_file, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            data.append(row)
    
    final_data = []
    for doc in tqdm(data, leave=False):
        #preprocess data
        continue
    
    return final_data


def main(config_path):
    config = Box.from_yaml(open(config_path, 'r'))
    input_data = read_data(config.data.data_dir, config.data.dataset)
    
    save_dir = Path(config.data.save_dir) / config.data.dataset
    save_dir.mkdir(exist_ok=True, parents=True)

    pickle.dump(input_data, open(save_dir / "data.pkl", 'wb'))
    

if __name__ == "__main__":
    args = parse_args()
    main(**args)
