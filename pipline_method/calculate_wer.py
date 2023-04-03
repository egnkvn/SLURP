import jiwer
import json

# 讀入JSON資料
with open('../datasets/slurp/slurp.json', 'r') as f:
    dataset = json.load(f)

train_data = dataset['train']
test_data = dataset['test']

for data in train_data:
  golden = data['golden']
  # # 計算WER
  if('google' in data):
    hypo = data['google']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['google']['wer'] = wer
  if('wav2vec2' in data):
    hypo = data['wav2vec2']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['wav2vec2']['wer'] = wer

for data in test_data:
  golden = data['golden']
  # # 計算WER
  if('google' in data):
    hypo = data['google']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['google']['wer'] = wer
  if('wav2vec2' in data):
    hypo = data['wav2vec2']['sentence']
    wer = jiwer.wer(golden, hypo)
    data['wav2vec2']['wer'] = wer

# # 將更新後的JSON寫回檔案中
with open('./slurp.json', 'w') as f:
    json.dump(dataset, f, indent=2)
