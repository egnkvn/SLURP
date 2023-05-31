import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./audio_training_dynamics.json')
    parser.add_argument('--save_path', type=str, default='./e2e_training_dynamics.png')
    parser.add_argument('--title', type=str, default='E2E (subset)')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.data_path) as json_file:
        data = json.load(json_file)
    
    sns.set()
    datamap = pd.DataFrame(data)
    print(len(datamap))

    pal = sns.diverging_palette(260, 15, n=len(datamap["correct_means"].unique().tolist()), sep=10, center="dark")

    fig = plt.figure(figsize=(14, 10), )
    sns.scatterplot(x="gold_prob_stds", y="gold_prob_means", data=datamap, palette=pal, hue="correct_means", s=10)

    plt.title(f'{args.title}')
    plt.savefig(f'{args.save_path}')

if __name__ == '__main__':
    main(parse_arguments())