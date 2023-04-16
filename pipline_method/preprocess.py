import jiwer
import json

def get_key(dic, val):
    for key, value in dic.items():
        if val == value:
            return key

# Read json data
with open('../datasets/slurp/slurp.json', 'r') as f:
    dataset = json.load(f)
with open('./meta.json', 'r') as f:
    meta = json.load(f)

train_data = dataset['train']
test_data = dataset['test']


for data in train_data:

  # Concat label
  s2id = meta['s2id']
  a2id = meta['a2id']
  intent2id = meta['intent2id']
  scenario = get_key(s2id, data['scenario'])
  action = get_key(a2id, data['action'])
  intent = scenario + '_' + action
  data['intent'] = meta['intent2id'][intent]

  # Calculate WER
  golden = data['golden']
  if('google' in data):
    hypo = data['google']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['google']['wer'] = wer
  if('wav2vec2' in data):
    hypo = data['wav2vec2']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['wav2vec2']['wer'] = wer


for data in test_data:

  # Concat label
  s2id = meta['s2id']
  a2id = meta['a2id']
  intent2id = meta['intent2id']
  scenario = get_key(s2id, data['scenario'])
  action = get_key(a2id, data['action'])
  intent = scenario + '_' + action
  data['intent'] = meta['intent2id'][intent]

  # Calculate WER
  golden = data['golden']
  if('google' in data):
    hypo = data['google']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['google']['wer'] = wer
  if('wav2vec2' in data):
    hypo = data['wav2vec2']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['wav2vec2']['wer'] = wer

# Write back to json
with open('./slurp.json', 'w') as f:
    json.dump(dataset, f, indent=2)
