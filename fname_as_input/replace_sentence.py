import json
import os
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../pipline_method/datasets/google/')
    parser.add_argument('--save_dir', type=str, default='./dataset')
    args = parser.parse_args()
    return args

def main(args):
    splits = ['train', 'valid']
    
    for split in splits:
        with open(os.path.join(args.data_dir, f'{split}.json')) as json_file:
            train_dataset = json.load(json_file)
        
        for obj in train_dataset:
            obj['sentence'] = obj['file']
        
        with open(os.path.join(args.save_dir, f'{split}_fname.json'), 'w') as writer:
            writer.write(json.dumps(train_dataset, indent=4))

if __name__ == '__main__':
    main(parse_arguments())