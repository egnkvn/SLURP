import json
import numpy as np 
import os
import pandas as pd
from argparse import ArgumentParser
from matplotlib_venn import venn2
from matplotlib import pyplot as plt

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--e2e_path', type=str, default='./e2e_method/datamap_labels.json')
    parser.add_argument('--pipeline_path', type=str, default='./pipline_method/datamap_labels.json')
    parser.add_argument('--save_path', type=str, default='./error_analysis.json')
    args = parser.parse_args()
    return args

def main(args):
    e2e_json = pd.read_json(args.e2e_path)
    pipeline_json = pd.read_json(args.pipeline_path)
    
    # Hard examples in E2E method might due to label error
    # Hard examples in two-stage method might be label error and ASR-error
    e2e_hard = set(e2e_json['hard'])
    pipeline_hard = set(pipeline_json['hard'])
    print(f'Number of hard examples in E2E method: {len(e2e_json["hard"])}')
    print(f'Number of hard examples in pipeline method: {len(pipeline_json["hard"])}')

    label_err = e2e_hard
    asr_err = pipeline_hard.difference(e2e_hard)

    err = {
        'label_err': list(label_err),
        'asr_err': list(asr_err)
    }
    with open(args.save_path, 'w') as out_file:
        out_file.write(json.dumps(err, indent=4))

    print(f'Number of examples with label error: {len(err["label_err"])}')
    print(f'Number of examples with ASR error: {len(err["asr_err"])}')

    venn2(
        subsets=[e2e_hard, pipeline_hard],
        set_labels=['E2E', 'Pipeline'],
        set_colors=['red', 'blue']
    )
    plt.savefig('./error_venn.png')
    
    return

if __name__ == '__main__':
    main(parse_arguments())
