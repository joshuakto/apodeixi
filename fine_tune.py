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


class Batch:
	def __init__(self, data_length, size):
		self.data_length = data_length
		self.size=size
		self.i=0
		self.z=0
		self.N=data_length//size+1
		self.n=0
		self.src=[]
		self.pred=None

class Metric:
	def __init__(self, df, args, writer):
		"""initialize array to store evaluations"""
		self.losses = np.zeros(args.evalsize)
		self.accu = np.zeros(args.evalsize)
		"""extract number of sources from file name and handle multi source data"""
		if not args.singlesource:
			srces = set(df.dsource)
			self.src_count = len(srces)
			self.src_accu = {src:np.zeros(args.evalsize) for src in srces}
			self.run_src_accu=0
		self.run_loss=0
		self.run_accu=0
		self.dig_labels=None
		self.loss=0
	
def init_parser():	
	"""data path setting"""
	parser = argparse.ArgumentParser(description='Fine tune roberta with stress tests.')
	test_data_path = '/home/bellman/nlp/roberta/data/MultiNLI_StressTest_Evaluation.jsonl'
	parser.add_argument('--datadir', default=test_data_path)
	parser.add_argument('--singlesource', action='store_true',
		help='enable when using dataset with single source(not mixed by datascrambler.py)')
	parser.add_argument('--batchsize', default=1)
	parser.add_argument('--evalsize', default=100,
		help='size of running loss/accu np array (to average evaluation metrics)')
	parser.add_argument('--loss', default='CrossEntropyLoss',
		help='choose between different loss functions: CrossEntroptLoss, NLLLoss, etc.')
	parser.add_argument('--expdir', default='tuned_model', 
		help='directory for tuned models from experiment')
	parser.add_argument('--summarycomment', default='', 
		help='add string to append to tensorboardX summary writer autonamed file')
	parser.add_argument('--incorrect_dir', default='mismatched_prediction/', 
		help='csv file for storing incorrectly predicted sentences')
	parser.add_argument('--nograd', action='store_true', 
		help='enable to disable training e.g. to evaluate model')
	parser.add_argument('--nocuda', action='store_true', 
		help='disable use of GPU')
	args = parser.parse_args()
	return args
		
def initialize_logging(args, df):
	"""log data to tensorboard"""
	writer = SummaryWriter(comment=args.summarycomment)
	metric = Metric(df,args, writer)
	"""create directory for storing finetuned model"""
	savemodeldir = os.path.join(args.expdir, args.summarycomment)
	if not os.path.exists(savemodeldir):
		os.mkdir(savemodeldir)
	if not os.path.exists(args.incorrect_dir):
		os.mkdir(args.incorrect_dir)
	return metric, writer, savemodeldir

def preprocess_jsonl(datadir):
	df = pd.read_json(datadir,lines=True)
	"""
	shuffle df rows refer to 
	https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
	"""
	df = df.sample(frac=1).reset_index(drop=True)  
	df.insert(0, 'batch_number', df.index)
	return df
	
def load_batch(df, batch):
	"""
	if loop to handle last batch (smaller than args.batchsize)
	"""
	if batch.n == batch.N-1:
		batch.i = batch.n*batch.size
		batch.z = len(df) - 1
	else:
		batch.i = batch.n*batch.size
		batch.z = batch.i+batch.size-1
	assert batch.z <= len(df), 'batch idx exceeds datafile size'
	"""
	MUST USE df.loc[i:z] (indexing different from df[i:z])for indexing see 
	https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
	"""
	sen1 = df.loc[batch.i:batch.z, 'sentence1'].values
	sen2 = df.loc[batch.i:batch.z, 'sentence2'].values
	sen_pairs = [[s1,s2] for s1, s2 in zip(sen1, sen2)]
	if not args.singlesource:
		batch.src = df.loc[batch.i:batch.z,'dsource'].to_list()
	return batch, sen_pairs

def extract_labels(df, batch, label_map, one_hot, device):
	labels = df.loc[batch.i:batch.z, 'gold_label'].values
	dig_labels = [label_map[l] for l in labels]
	dig_labels = torch.tensor(dig_labels, dtype = torch.long)
	if one_hot:
		index_holder = torch.tensor([[dig_labels]], dtype=torch.long)
		dig_labels = torch.LongTensor(batch.size, 3).zero_()
		dig_labels.scatter_(1, index_holder, 1)
	dig_labels = dig_labels.to(device=device)
	return dig_labels
"""
	sample incorrect predictions, with information on:
		the batch#
		data source (which corpus)
"""

"""
	TODO:
		Fix inconsistency between in treatment of batch (batch>1 will break the code)
		embed more info in mixed data using datascrambler.py
"""
def evaluate_prediction(args, df, batch, metric, writer, dig_labels):
	writer.add_scalar('loss/loss', metric.loss, batch.n)
	metric.losses = push_np_array(metric.losses, metric.loss)
	try:
	        accubin = [int(e) for e in torch.eq(batch.pred,dig_labels)]
	except:
	        accubin = [int(e) for e in torch.eq(batch.pred,dig_labels.argmax(dim=1))]
	metric.accu = push_np_array(metric.accu, accubin)
	if not args.singlesource:
		for src in batch.src:
			metric.src_accu[src] = push_np_array(metric.src_accu[src], accubin)
	if batch.n%args.evalsize==0:
	        metric = average_evaluation(metric)
	        gen_report(batch, metric, writer, dig_labels)
	if accubin == 0:
		inc_sen_p = os.path.join( 
			args.incorrect_dir,args.summarycomment+
			'_incorrect_prediction.jsonl')
		with open(inc_sen_p, 'a') as of:
			of.write(df.loc[batch.i:batch.z].to_json(orient='records',lines=True))
			of.write('\n')
	return metric



def push_np_array(l, n):
	try:
		l = np.roll(l, len(n))
		l[:len(n)] = n
	except:
		l = np.roll(l,1)
		l[0]=n
	return l

def average_evaluation(metric):
	metric.run_loss = np.average(metric.losses)
	metric.run_accu = np.sum(metric.accu)
	if not args.singlesource:
		metric.run_src_accu = {src:metric.src_accu[src].sum() 
			for src in metric.src_accu.keys()}
	return metric

def gen_report(batch, metric, writer, dig_labels):
	try:
		dig_labels = dig_labels.tolist()
	except:
		dig_labels = dig_labels.argmax(dim=1).tolist()
	print(
		'Batch{}| running loss:{}|'.format(batch.n, metric.run_loss)+
		' accu:{}| src_accu:{}'.format(metric.run_accu, metric.run_src_accu if not args.singlesource else 'NA')+
		' prediction:{}| ground truth:{}'.format(batch.pred.tolist(), dig_labels)
	)
	writer.add_scalar('loss/running_loss', metric.run_loss, batch.n)
	writer.add_scalar('accu/accuracy', metric.run_accu, batch.n)
	if not args.singlesource:
		writer.add_scalars('accu/src_accuracy',metric.run_src_accu, batch.n)
	return

if __name__ == '__main__':
	args = init_parser()
	df = preprocess_jsonl(args.datadir)

	"""model settings and training settings(loss function and optimizer)"""	
	grad = not(args.nograd)
	run_cuda = not(args.nocuda)
	roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
	device='cpu'
	if run_cuda:
		roberta.cuda()
		device='cuda'
	if grad:
		roberta.train()
		optimizer = optim.Adam(roberta.parameters(), lr=1e-8)
	else:
		roberta.eval()
	criterion_map = {
		'CrossEntropyLoss':nn.CrossEntropyLoss(),
		'NLLLoss':nn.NLLLoss(),
		'MultiLabelMarginLoss':nn.MultiLabelMarginLoss()
	}
	criterion = criterion_map[args.loss]
	one_hot_loss = ['MultiLabelMarginLoss'] 
	onehot = args.loss in one_hot_loss
	batch = Batch(len(df),int(args.batchsize)) 
	"""
	initialize logging arrays, writer and directories
	"""
	(metric, writer, savemodeldir) = initialize_logging(args, df)

	label_map = {'contradiction':0, 'neutral':1, 'entailment':2}
	for batch.n in range(batch.N):
		batch, sen_pairs= load_batch(df, batch)
		try:
			batch_tokens = collate_tokens(
				[roberta.encode(pair[0], pair[1])
				for pair in sen_pairs], pad_idx=1)
			logprobs = roberta.predict('mnli', batch_tokens)
			batch.pred = logprobs.argmax(dim=1)
			logprobs = logprobs.to(device=device)
			"""
			Update model using loss function
			"""
			with torch.set_grad_enabled(grad):
				dig_labels = extract_labels(df, batch, 
						label_map, onehot, device)
				metric.loss = criterion(logprobs, dig_labels)
				if grad:
					metric.loss.backward()
					optimizer.step()
				metric=evaluate_prediction(args, df, batch,
					metric, writer, dig_labels)
		#except ValueError as err:
		#	print('Batch{}:index[{}:{}]| ValueError:{}'.format(
		#		batch.n, batch.i, batch.z, err))	
		except KeyError as err:
			print('Batch{}:index[{}:{}] missing gold label| KeyError:{}'
				.format(batch.n, batch.i, batch.z, err))
		#except:
		#	print('Unexpected error:', sys.exc_info()[:2])
		if batch.n%10000 == 0 or batch.n==batch.N-1:
			torch.save(roberta.state_dict(),'{}.pt'.format(
				os.path.join(savemodeldir, str(batch.n))))
	writer.close()
	"""
	writing graph using pytorch native summary writer
	refer to:
	https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#tensorboard-setup
	"""
	# from torch.utils.tensorboard import PytSummaryWriter
	# writer2 = PytSummaryWriter()
	# tk = roberta.encode('dummy')
	# dummy = roberta.extract_features(tk)
	# writer2.add_graph(roberta.model.classification_heads['mnli'], dummy)
