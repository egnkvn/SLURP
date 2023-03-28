import os
import pandas as pd
import shutil
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--audio_dir', type=str, default='./audio/slurp_real')
    parser.add_argument('--annotation_dir', type=str, default='./slurp')
    args = parser.parse_args()
    return args

def main(args):
    filenames = {
        'train': list(),
        'devel': list(),
        'test': list()
    }
    
    # Get the audio file names of each split (from annotation data).
    for split in filenames.keys():
        json_obj = pd.read_json(os.path.join(args.annotation_dir, f'{split}.jsonl'), lines=True)
        for idx in range(len(json_obj)):
            recordings = json_obj['recordings'][idx]
            for i in range(len(recordings)):
                if recordings[i]['file'] not in filenames[split]:
                    filenames[split].append(recordings[i]['file'])
    for split in filenames.keys():
        print(len(filenames[split]))

    # Assert training set is disjoin with devel set and testing set
    for fname in filenames['train']:
        if fname in filenames['test']:
            print(f'{fname} exists in train and test')
        if fname in filenames['devel']:
            print(f'{fname} exists in train and devel')

    # Move the training set data to the corresponding dir
    for fname in filenames['train']:
        shutil.move(os.path.join(args.audio_dir, fname), os.path.join(args.audio_dir, 'train', fname))
    
    train_files = os.listdir(os.path.join(args.audio_dir, 'train'))
    for fname in filenames:
        if fname not in train_files:
            print(f'{fname} is not moved')
    
    return

if __name__ == '__main__':
    main(parse_arguments())