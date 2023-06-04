import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./training_dynamic/golden.json')
    parser.add_argument('--save_path', type=str, default='./golden.png')
    parser.add_argument('--title', type=str, default='Pipeline Method (golden)')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.data_path) as json_file:
        data = json.load(json_file)
    
    datamap = {
        'confidence': [data[fname]['prob_mean'] for fname in data.keys()],
        'variability': [data[fname]['prob_std'] for fname in data.keys()],
        'correct': [data[fname]['correct_mean'] for fname in data.keys()]
    }
    datamap = pd.DataFrame(datamap)

    sns.set()
    pal = sns.diverging_palette(260, 15, n=len(datamap["correct"].unique().tolist()), sep=10, center="dark")

    fig = plt.figure(figsize=(14, 10), )
    sns.scatterplot(x="variability", y="confidence", data=datamap, palette=pal, hue="correct", s=10)

    plt.title(f'{args.title}')
    plt.savefig(f'{args.save_path}')

if __name__ == '__main__':
    main(parse_arguments())