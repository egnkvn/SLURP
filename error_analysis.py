import json
import numpy as np 
import os
import pandas as pd
from argparse import ArgumentParser
from matplotlib_venn import venn2, venn3
from matplotlib import pyplot as plt

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--e2e_path', type=str, default='./e2e_method/datamap_labels.json')
    parser.add_argument('--pipeline_path', type=str, default='./pipline_method/training_dynamic/analysis/asr.json')
    parser.add_argument('--golden_text_path', type=str, default='./pipline_method/training_dynamic/analysis/golden.json')
    parser.add_argument('--save_path', type=str, default='./error_analysis.json')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.e2e_path) as json_file:
        e2e_json = json.load(json_file)
    pipeline_json = pd.read_json(args.pipeline_path)
    with open(args.golden_text_path) as json_file:
        golden_text_json = json.load(json_file)
    
    # Hard examples in E2E method might due to label error
    # Hard examples in two-stage method might be label error and ASR-error
    e2e_hard = set(e2e_json['hard'])
    pipeline_hard = set(pipeline_json['hard'])
    golden_text_hard = set(golden_text_json['hard'])
    print(f'Number of hard examples in E2E method: {len(e2e_json["hard"])}')
    print(f'Number of hard examples in pipeline method (ASR): {len(pipeline_json["hard"])}')
    print(f'Number of hard examples in pipeline method (golden text): {len(pipeline_json["hard"])}')

    purple = e2e_hard.intersection(pipeline_hard)
    green = pipeline_hard.intersection(golden_text_hard)
    orange = golden_text_hard.intersection(e2e_hard)

    err = dict()
    err['pink'] = list(e2e_hard.intersection(pipeline_hard, golden_text_hard))  # E2E, ASR, Golden
    err['purple'] = list(purple.difference(err['pink']))                        # E2E, ASR
    err['green'] = list(green.difference(err['pink']))                          #      ASR, Golden
    err['orange'] = list(orange.difference(err['pink']))                        # E2E,      Golden
    err['red'] = list(e2e_hard.difference(purple, orange))                      # E2E
    err['yellow'] = list(golden_text_hard.difference(orange, green))            #           Golden
    err['blue'] = list(pipeline_hard.difference(green, purple))                 #       ASR

    for k, v in err.items():
        print(f'key = {k}, length = {len(v)}')

    
    with open(args.save_path, 'w') as out_file:
        out_file.write(json.dumps(err, indent=4))

    venn3(
        subsets=[e2e_hard, pipeline_hard, golden_text_hard],
        set_labels=['E2E', 'Pipeline (Google-ASR)', 'Pipeline (golden text)'],
        set_colors=['red', 'blue', 'yellow']
    )
    plt.savefig('./error_venn.png')
    
    return

if __name__ == '__main__':
    main(parse_arguments())
