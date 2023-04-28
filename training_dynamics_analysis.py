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
    args = parser.parse_args()
    return args

def relabel(training_dynamics, file_names):
    var_top33 = np.percentile(training_dynamics["gold_prob_stds"], 67)      # Ambiguous
    conf_btm33 = np.percentile(training_dynamics["gold_prob_means"], 33)    # Hard
    conf_top33 = np.percentile(training_dynamics["gold_prob_means"], 67)    # Easy

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
        pass                                                           
    print(f'Number of files: {len(file_names)}')

    # Relabel examples
    relabeled_data = relabel(training_dynamics, file_names)

    # Dump file
    with open(args.save_path, 'w') as out_file:
        out_file.write(json.dumps(relabeled_data, indent=4))

    

    return

if __name__ == '__main__':
    main(parse_arguments())