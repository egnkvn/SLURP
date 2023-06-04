import json
import random
import numpy as np

with open('../fname_as_input/fname.json', 'r') as f:
    fname = json.load(f)
with open('./training_dynamic/record.json', 'r') as f:
    record = json.load(f)

fname = fname["fname"]
prob = record["prob"]
correct = record["correct"]
total_instances = len(fname)

result = dict()
for id in range(total_instances):
  File = fname[id]
  if(File in result):
    result[File]["prob"].append(prob[id])
    result[File]["correct"].append(correct[id])
  else:
    result[File] = {
      "prob": [prob[id]],
      "correct": [correct[id]]
    }

output = dict()
for File in result:
  prob_mean = np.mean(np.array(result[File]["prob"]))
  prob_std = np.std(np.array(result[File]["prob"]))
  correct_mean = np.mean(np.array(result[File]["correct"]))
  output[File] = {
    "prob_mean": prob_mean,
    "prob_std": prob_std,
    "correct_mean": correct_mean
  }

with open('./training_dynamic/golden.json', 'w') as f:
    json.dump(output, f, indent=2)


  
