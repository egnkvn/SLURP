import jiwer
import json
import copy

def get_key(dic, val):
  for key, value in dic.items():
    if val == value:
      return key

def append2array(array, data, src, rm):
  tmp = {}
  tmp['sentence'] = ''
  # tmp = copy.deepcopy(data)
  for key in data:
    tmp[key] = data[key]
  if(rm in tmp):
    del tmp[rm]
  tmp['sentence'] = data['golden']
  tmp['wer'] = data[src]['wer']
  del tmp[src]
  array.append(tmp)

# Read json data
with open('./datasets/slurp.json', 'r') as f:
    dataset = json.load(f)
with open('./datasets/meta.json', 'r') as f:
    meta = json.load(f)

train_data = dataset['train']
val_data = dataset['devel']
test_data = dataset['test']

google_train = []
google_val= []
google_test = []
w2v2_train = []
w2v2_val= []
w2v2_test = []

for data in train_data:

  # Concat label
  s2id = meta['s2id']
  a2id = meta['a2id']
  intent2id = meta['intent2id']
  scenario = get_key(s2id, data['scenario'])
  action = get_key(a2id, data['action'])
  data['label'] = scenario + '_' + action

  # Calculate WER
  golden = data['golden']
  if('google' in data):
    hypo = data['google']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['google']['wer'] = wer
    append2array(google_train, data, 'google', 'wav2vec2')
  if('wav2vec2' in data):
    hypo = data['wav2vec2']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['wav2vec2']['wer'] = wer
    append2array(w2v2_train, data, 'wav2vec2', 'google')

for data in val_data:

  # Concat label
  s2id = meta['s2id']
  a2id = meta['a2id']
  intent2id = meta['intent2id']
  scenario = get_key(s2id, data['scenario'])
  action = get_key(a2id, data['action'])
  data['label'] = scenario + '_' + action

  # Calculate WER
  golden = data['golden']
  if('google' in data):
    hypo = data['google']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['google']['wer'] = wer
    append2array(google_val, data, 'google', 'wav2vec2')
  if('wav2vec2' in data):
    hypo = data['wav2vec2']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['wav2vec2']['wer'] = wer
    append2array(w2v2_val, data, 'wav2vec2', 'google')

for data in test_data:

  # Concat label
  s2id = meta['s2id']
  a2id = meta['a2id']
  intent2id = meta['intent2id']
  scenario = get_key(s2id, data['scenario'])
  action = get_key(a2id, data['action'])
  data['label'] = scenario + '_' + action

  # Calculate WER
  golden = data['golden']
  if('google' in data):
    hypo = data['google']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['google']['wer'] = wer
    append2array(google_test, data, 'google', 'wav2vec2')
  if('wav2vec2' in data):
    hypo = data['wav2vec2']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['wav2vec2']['wer'] = wer
    append2array(w2v2_test, data, 'wav2vec2', 'google')

# Write back to json
with open('./datasets/google/train_golden.json', 'w') as f:
    json.dump(google_train, f, indent=2)

with open('./datasets/google/valid_golden.json', 'w') as f:
    json.dump(google_val, f, indent=2)

with open('./datasets/google/test_golden.json', 'w') as f:
    json.dump(google_test, f, indent=2)

with open('./datasets/wav2vec2/train_golden.json', 'w') as f:
    json.dump(w2v2_train, f, indent=2)

with open('./datasets/wav2vec2/valid_golden.json', 'w') as f:
    json.dump(w2v2_val, f, indent=2)

with open('./datasets/wav2vec2/test_golden.json', 'w') as f:
    json.dump(w2v2_test, f, indent=2)
