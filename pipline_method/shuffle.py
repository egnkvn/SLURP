import json
import random

with open('datasets/google/train_golden.json', 'r') as file:
    data = json.load(file)

random.shuffle(data)

with open('./datasets/google/train_golden_v2.json', 'w') as f:
    json.dump(data, f, indent=2)