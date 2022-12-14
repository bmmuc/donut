from donut import DonutModel
from PIL import Image
pretrained_model = DonutModel.from_pretrained('/home/ec2-user/donut/result/test')
import json
import ast
import numpy as np
meta = []
import torch
path_base = '/home/ec2-user/donut/dataset/v1/test'
with open(path_base, 'r', encoding='utf-8') as f:
    for line in f:
        meta.append(json.loads(line))
new_meta = list()

for ds in meta:
  new_meta_ = dict()
  new_meta_['file_name'] = ds['file_name']
  gt = ast.literal_eval(ds['ground_truth'])['gt_parses']
  new_meta_['question_1'] = gt[0]['question']
  new_meta_['answer_1'] = gt[0]['answer']
  new_meta_['question_2'] = gt[1]['question']
  new_meta_['answer_2'] = gt[1]['answer']
  new_meta_['question_3'] = gt[2]['question']
  new_meta_['answer_3'] = gt[2]['answer']  
  new_meta_['question_4'] = gt[3]['question']
  new_meta_['answer_4'] = gt[3]['answer']
  new_meta.append(new_meta_)

import time
if torch.cuda.is_available():
    pretrained_model.half()
    pretrained_model.to("cuda")

pretrained_model.eval()

questions = ['What is the cpf or cnpj of the prestador?',
             'What is the name of the prestador',
             'What is the cpf or cnpj of the tomador?',
             'What is the descriminação de serviço?',
             ]

predictions = []
ground_truths = []
accs_cnpj = []
accs_names = []
accs_cpf = []
accs_desc = []

times = []
for ds in meta:
  img = Image.open(f'{path_base}/{ds["file_name"]}')

  a = time.time()
  output = pretrained_model.inference(
      image=img,
      prompt=f"<s_docvqa><s_question>{ds['question_1']}</s_question><s_answer>")
  b = time.time() - a

  times.append(b)
  if 'answer' not in output[0]['predictions'].keys():
    print(output)
    continue
  acc_1 = float(output['predictions'][0]['answer'] == ds['answer_1'])
  accs_cnpj.append(acc_1)

  a = time.time()
  output = pretrained_model.inference(
      image=img,
      prompt=f"<s_docvqa><s_question>{ds['question_2']}</s_question><s_answer>")
  b = time.time() - a
  times.append(b)
  if 'answer' not in output[0]['predictions'].keys():
    print(output)
    continue
  acc_1 = float(output['predictions'][0]['answer'] == ds['answer_2'])
  accs_names.append(acc_1)

  a = time.time()
  output = pretrained_model.inference(
      image=img,
      prompt=f"<s_docvqa><s_question>{ds['question_2']}</s_question><s_answer>")
  b = time.time() - a
  times.append(b)
  if 'answer' not in output[0]['predictions'].keys():
    print(output)
    continue
  acc_1 = float(output['predictions'][0]['answer'] == ds['answer_2'])
  accs_cpf.append(acc_1)
  a = time.time()

  output = pretrained_model.inference(
      image=img,
      prompt=f"<s_docvqa><s_question>{ds['question_4']}</s_question><s_answer>")
  b = time.time() - a
  times.append(b)
  if 'answer' not in output[0]['predictions'].keys():
    print(output)
    continue

  acc_1 = float(output['predictions'][0]['answer'] == ds['answer_4'])
  accs_cpf.append(acc_1)

print(np.mean(times), np.mean(accs_cnpj), np.mean(accs_names))