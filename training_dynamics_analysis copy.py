import json
import numpy as np 
import os
import pandas as pd
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--method', type=str)                               # e2e or pipeline
    parser.add_argument('--training_dynamics_path', type=str)               # The file recording training dynamics
    parser.add_argument('--filename_info', type=str)                        # Where to get file names
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--ambig_thres', type=int, default=67)
    parser.add_argument('--hard_thres', type=int, default=33)
    parser.add_argument('--easy_thres', type=int, default=67)
    args = parser.parse_args()
    return args

def relabel(training_dynamics, file_names, ambig_thres, hard_thres, easy_thres):
    var_top33 = np.percentile(training_dynamics["gold_prob_stds"], ambig_thres)     # Ambiguous
    conf_btm33 = np.percentile(training_dynamics["gold_prob_means"], hard_thres)    # Hard
    conf_top33 = np.percentile(training_dynamics["gold_prob_means"], easy_thres)    # Easy

    relabeled_data = {
        'ambig': list(),
        'hard': list(),
        'easy': list()
    }

    for idx in range(len(file_names)):
        if training_dynamics["gold_prob_stds"][idx] >= var_top33:           # Ambiguous
            relabeled_data['ambig'].append(file_names[idx])
        if training_dynamics["gold_prob_means"][idx] <= conf_btm33:         # Hard
            relabeled_data['hard'].append(file_names[idx])
        if training_dynamics["gold_prob_means"][idx] >= conf_top33:         # Easy
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
    if args.method == 'e2e':
        df = pd.read_csv(args.filename_info)
        file_names = [fname.split('data/')[1] for fname in df['file_name']]
    else:
        # TODO
        df = pd.read_json(args.filename_info)
        # file_names = [fname.split('data/')[1] for fname in df['file']]
        print(df['file'][0])
        file_names = [fname for fname in df['file']]
        pass                                                           
    print(f'Number of files: {len(file_names)}')

    # Relabel examples
    relabeled_data = relabel(training_dynamics, file_names, args.ambig_thres, args.hard_thres, args.easy_thres)

    # Dump file
    with open(args.save_path, 'w') as out_file:
        out_file.write(json.dumps(relabeled_data, indent=4))

    return

if __name__ == '__main__':
    main(parse_arguments())