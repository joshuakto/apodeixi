import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tensorboardX import SummaryWriter
import time
import sys
sys.path.append('/home/bellman/nlp/roberta/fairseq')
from fairseq.data.data_utils import collate_tokens
import os
import numpy as np

def push_np_array(l, n):
	l = np.roll(l, 1)
	l[0] = n
	return l

def evaluation(losses, accuracies, steps):
	running_loss = np.average(losses)
	running_accu = np.sum(accuracies)	
	return running_loss, running_accu

def gen_report(batchnumber, running_loss, accuracy, pred, glabel):
	print('Batch{}| running loss:{}| accu:{}| prediction:{}| ground truth:{}'.format(
		batchnumber, running_loss, accuracy, pred, glabel)
		)
	return

def preprocess_jsonl(datadir):
	df = pd.read_json(datadir,lines=True)
	# shuffle df rows refer to https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
	df = df.sample(frac=1).reset_index(drop=True)  
	return df
	
def load_batch(df, Number_of_Batches, batch_number, batchsize):
	# if loop to handle last batch (smaller than args.batchsize)
	if batch_number == Number_of_Batches-1:
		batch_i = batch_number*batchsize
		batch_z = len(df) - 1
	else:
		batch_i = batch_number*batchsize
		batch_z = batch_i+batchsize-1
	assert batch_z <= len(df), 'batch idx exceeds datafile size'
	# MUST USE df.loc[i:z] (indexing different from df[i:z])for indexing see 
	# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
	sen1 = df.loc[batch_i:batch_z, 'sentence1'].values
	sen2 = df.loc[batch_i:batch_z, 'sentence2'].values
	sen_pairs = [[s1,s2] for s1, s2 in zip(sen1, sen2)]
	return batch_i, batch_z, sen_pairs

if __name__ == '__main__':
	# data path setting
	parser = argparse.ArgumentParser()
	test_data_path = '/home/bellman/nlp/roberta/data/MultiNLI_StressTest_Evaluation.jsonl'
	parser.add_argument('--datadir', default=test_data_path)
	parser.add_argument('--batchsize', default=1)
	parser.add_argument('--evalsize', default=100)
	parser.add_argument('--nograd', action='store_true')
	args = parser.parse_args()
	df = preprocess_jsonl(args.datadir)
	# model settings and training settings(loss function and optimizer)
	nograd = args.nograd
	roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
	if nograd:
		roberta.eval()
	else:
		roberta.train()
	roberta.cuda()
	criterion = nn.CrossEntropyLoss()
	########### CHANGE optim hyperparam here ############
	if not(nograd):
		optimizer = optim.Adam(roberta.parameters(), lr=1e-8)
	# initialize array to store evaluations
	losses = np.zeros(args.evalsize)
	accuracies = np.zeros(args.evalsize)
	# log data to tensorboard
	writer = SummaryWriter(comment='Adam_low_lr')

	Number_of_Batches = len(df)//args.batchsize+1
	s = time.time()
	label_map = {'contradiction':0, 'neutral':1, 'entailment':2}
	for batch_number in range(Number_of_Batches):
		batch_i, batch_z, sen_pairs = load_batch(
			df, Number_of_Batches, batch_number, args.batchsize)
		try:
			batch = collate_tokens(
				[roberta.encode(pair[0], pair[1])
				for pair in sen_pairs], pad_idx=1)
		except:
			pass
		logprobs = roberta.predict('mnli', batch)
		pred = logprobs.argmax(dim=1)

		# Update model using loss function
		with torch.set_grad_enabled(not(nograd)):
			try:
				labels = df.loc[batch_i:batch_z, 'gold_label'].values
				dig_labels = [label_map[l] for l in labels]
				dig_labels = torch.tensor(dig_labels, dtype = torch.long)
				dig_labels = dig_labels.to(device='cuda')
				logprobs = logprobs.to(device='cuda')
				loss = criterion(logprobs, dig_labels)
				if not(nograd):
					loss.backward()
					optimizer.step()
				writer.add_scalar('loss', loss, batch_number)
				losses = push_np_array(losses, loss)
				accubin = 1 if pred.item()==dig_labels else 0
				accuracies = push_np_array(accuracies, accubin)
				if batch_number%args.evalsize==0:	
					running_loss, running_accu = evaluation(
						losses, accuracies, args.evalsize)	
					gen_report(batch_number, running_loss, running_accu,
						pred.item(), dig_labels.item())
					writer.add_scalar('running_loss', running_loss, batch_number)
					writer.add_scalar('accuracy', running_accu, batch_number)
			except KeyError:
				print('Batch{}:index[{}:{}] missing gold label'
					.format(batch_number, batch_i, batch_z))
	writer.close()
			
