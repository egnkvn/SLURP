import json
import numpy as np 
import os
import pandas as pd
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--training_dynamics_path', type=str)               # The file recording training dynamics
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--ambig_thres', type=float)
    parser.add_argument('--hard_thres', type=float, default=1)
    parser.add_argument('--easy_thres', type=float)
    args = parser.parse_args()
    return args

def relabel(training_dynamics, file_names, ambig_thres, hard_thres, easy_thres):
    var_top33 = np.percentile(training_dynamics["prob_std"], ambig_thres)     # Ambiguous
    conf_btm33 = np.percentile(training_dynamics["prob_mean"], hard_thres)    # Hard
    conf_top33 = np.percentile(training_dynamics["prob_mean"], easy_thres)    # Easy

    print(f'Ambiguous threshold: Variability > {var_top33}')
    print(f'Hard threshold: Confidence < {conf_btm33}')
    print(f'Easy threshold: Confidence > {conf_top33}')

    relabeled_data = {
        'ambig': list(),
        'hard': list(),
        'easy': list()
    }

    for idx in range(len(file_names)):
        if training_dynamics["prob_std"][idx] >= var_top33:           # Ambiguous
            relabeled_data['ambig'].append(file_names[idx])
        if training_dynamics["prob_mean"][idx] <= conf_btm33:         # Hard
            relabeled_data['hard'].append(file_names[idx])
        if training_dynamics["prob_mean"][idx] >= conf_top33:         # Easy
            relabeled_data['easy'].append(file_names[idx])

    print(f'Number of ambiguous examples: {len(relabeled_data["ambig"])}')
    print(f'Number of hard examples: {len(relabeled_data["hard"])}')
    print(f'Number of easy examples: {len(relabeled_data["easy"])}')

    return relabeled_data

def main(args):
    # Load training dynamic records
    with open(args.training_dynamics_path) as json_file:
        training_dynamics = json.load(json_file)
        
    # Get file names
    file_names = list(training_dynamics.keys())
    training_dynamics = {
        'prob_mean':  [training_dynamics[fname]['prob_mean'] for fname in training_dynamics.keys()],
        'prob_std': [training_dynamics[fname]['prob_std'] for fname in training_dynamics.keys()]
    }
    print(f'Number of files: {len(file_names)}')

    # Relabel examples
    relabeled_data = relabel(training_dynamics, file_names, args.ambig_thres, args.hard_thres, args.easy_thres)

    # Dump file
    with open(args.save_path, 'w') as out_file:
        out_file.write(json.dumps(relabeled_data, indent=4))

    return

if __name__ == '__main__':
    main(parse_arguments())