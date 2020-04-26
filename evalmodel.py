import argparse
# work in progress from model.utils.data_u import 
import sys
import pandas as pd
import time
import torch
sys.path.append('/home/bellman/nlp/roberta/fairseq')
from fairseq.data.data_utils import collate_tokens
import numpy as np
import os

def push_np_array(l, n):
        l = np.roll(l, 1)
        l[0] = n
        return l

def extract_labels(df, batch_i, batch_z, label_map_r):
        labels = df.loc[batch_i:batch_z, 'gold_label'].values
        dig_labels = [label_map_r[l] for l in labels]
        dig_labels = torch.tensor(dig_labels, dtype = torch.long)
        return dig_labels


parser = argparse.ArgumentParser(description='Predict textual entailment using roberta.')
parser.add_argument('--data_jsonl', dest='data', 
	default='data/MultiNLI_StressTest_Evaluation.jsonl',
	help='File containing sentence pair data')
parser.add_argument('--insert_col', action='store_true')
parser.add_argument('--nocuda', action='store_true')
parser.add_argument('--altmodel', default=None)

args = parser.parse_args()
data = args.data
runs_cuda = not(args.nocuda)
device = 'cuda' if runs_cuda else 'cpu'

df = pd.read_json(data, lines = True)
if args.nocuda:
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if args.insert_col:
	df.insert(4,'label',None)
df.insert(4,'prediction',None)
label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
label_map_r = {'contradiction':0, 'neutral':1, 'entailment':2}

batch_size = 1
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
if not(args.altmodel==None):
	device = torch.device('cpu')
	roberta.load_state_dict(torch.load(args.altmodel, map_location=device))
	roberta.eval()
if not(args.nocuda):
	roberta.cuda()	
BatchNum = len(df)//batch_size + 1 
accuracies = np.zeros(100)
for batch in range(BatchNum):
	accu = np.sum(accuracies)
	print('Batch #{} | accu:{}'.format(batch, accu))
	if batch == BatchNum - 1:	
		batch_i = batch*batch_size
		batch_z = len(df) - 1
	else:
		batch_i = batch*batch_size
		batch_z = batch_i + batch_size - 1
	assert batch_z <= len(df), 'batch idx exceeds datafile size'	
	sen1 = df.loc[batch_i:batch_z, 'sentence1'].values
	sen2 = df.loc[batch_i:batch_z, 'sentence2'].values
	sen_pairs = [[s1,s2] for s1, s2 in zip(sen1, sen2)] 
	with torch.no_grad():
		try:
			bat = collate_tokens(
				[roberta.encode(pair[0], pair[1]) for pair in sen_pairs], pad_idx=1)
		except:
			import pprint as pp
			pp.pprint(sen_pairs)
			print(sen1)
			print(sen2)
			print(b)
			print(batch_i)
			print(batch_z)
			break
		logprobs = roberta.predict('mnli', bat) 
		labels = logprobs.argmax(dim=1).tolist()
		df.loc[batch_i:batch_z,'label'] = labels
		pred = logprobs.argmax(dim=1)
		dig_labels = extract_labels(df, batch_i, batch_z, label_map_r)
		accubin = 1 if pred.item()==dig_labels else 0
		accuracies = push_np_array(accuracies, accubin)
