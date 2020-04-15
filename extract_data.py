# from model.utils.batchClass import EntailBatch, Content
import pickle
import argparse
# work in progress from model.utils.data_u import 
import sys
import pandas as pd
import time
import torch
sys.path.append('/home/bellman/nlp/roberta/fairseq')
from fairseq.data.data_utils import collate_tokens

parser = argparse.ArgumentParser(description='Predict textual entailment using roberta.')
parser.add_argument('--data_jsonl', dest='data', 
	default='data/MultiNLI_StressTest_Evaluation.jsonl',
	help='File containing sentence pair data')
parser.add_argument('--insert_col', action='store_true')

args = parser.parse_args()
data = args.data

df = pd.read_json(data, lines = True)
if args.insert_col:
	df.insert(4,'label',None)
df.insert(4,'prediction',None)
label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}

batch_size = 100
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
# run on GPU
roberta.cuda()	
# Number of batches required (+1 for last incomplete batch)
BatchNum = len(df)//batch_size + 1 
s = time.time()
for batch in range(BatchNum):
	print('Batch #' + str(batch))
	# take care of last incomplete batch
	if batch == BatchNum - 1:	
		batch_i = batch*batch_size
		batch_z = len(df) - 1
	else:
		batch_i = batch*batch_size
		batch_z = batch_i + batch_size - 1
	assert batch_z <= len(df), 'batch idx exceeds datafile size'	
	# MUST USE df.loc[i:z] (indexing different from df[i:z])for indexing see 
	# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
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
		predictions = [label_map[l] for l in labels]
		df.loc[batch_i:batch_z, 'prediction'] = predictions
print('Time Taken: '+ str(time.time()-s))
# # Need to write a function to format it to sample_submission.jsonl
# # USE 'result_temp_storage.pkl' as the testing df 
# df.to_json('testingtojson.jsonl', orient = 'records', lines=True)
with open('result_temp_storage.pkl', 'wb') as wf:
	pickle.dump(df,wf)	

"""
================================Batch Pred Trial================================
================================13th April, 2020================================
"""

#df = pd.read_json('MultiNLI_StressTest_Evaluation.jsonl', lines = True)
#"""
#structure of sample_submission for eval.py on https://github.com/AbhilashaRavichander/NLI_StressTest
#['promptID', 'annotator_labels', 'sentence1_binary_parse', 'sentence2_parse', 'prediction, 'label', 'source', 'sentence2_binary_parse', 'pairID', 'sentence2', 'sentence1_parse', 'genre', 'gold_label', 'sentence1']
#Therefore, we need to add to original dataframe(['promptID', 'annotator_labels', 'sentence1_binary_parse', 'sentence2_parse', 'source', 'sentence2_binary_parse', 'pairID', 'sentence2', 'sentence1_parse', 'genre', 'gold_label', 'sentence1'])
#'prediciton' and 'label'
#For example: 'prediction':'contradiction', 'label':0
#Where label --> prediction:
#0 --> contradiction
#1 --> neutral
#2 --> entailment
#"""
#df.insert(4,'label',None)
#df.insert(4,'prediction',None)
#label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
#
#batch_size = 100
#roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
## run on GPU
#roberta.cuda()	
## Number of batches required (+1 for last incomplete batch)
#BatchNum = len(df)//batch_size + 1 
#s = time.time()
#for batch in range(BatchNum):
#	print('Batch #' + str(batch))
#	# take care of last incomplete batch
#	if batch == BatchNum - 1:	
#		batch_i = batch*batch_size
#		batch_z = len(df) - 1
#	else:
#		batch_i = batch*batch_size
#		batch_z = batch_i + batch_size - 1
#	assert batch_z <= len(df), 'batch idx exceeds datafile size'	
#	## DO NOT USE df.loc: indexing is different
#	## b = df.loc[batch_i:batch_z]
#	sen1 = df.sentence1[batch_i:batch_z].values
#	sen2 = df.sentence2[batch_i:batch_z].values
#	sen_pairs = [[s1,s2] for s1, s2 in zip(sen1, sen2)] 
#	with torch.no_grad():
#		try:
#			bat = collate_tokens(
#				[roberta.encode(pair[0], pair[1]) for pair in sen_pairs], pad_idx=1)
#		except:
#			import pprint as pp
#			pp.pprint(sen_pairs)
#			print(sen1)
#			print(sen2)
#			print(b)
#			print(batch_i)
#			print(batch_z)
#			break
#		logprobs = roberta.predict('mnli', bat) 
#		labels = logprobs.argmax(dim=1).tolist()
#		# batch_z + 1 to accomodate for difference between
#		# df.label[i:z + 1], df.prediction[i:z +1] indexing AND 
#		# df.loc[i:z] indexing
#		df.label[batch_i:batch_z] = labels
#		predictions = [label_map[l] for l in labels]
#		df.prediction[batch_i:batch_z] = predictions
#print('Time Taken: '+ str(time.time()-s))
#import pickle
#with open('result_temp_storage.pkl', 'wb') as wf:
#	pickle.dump(df,wf)	


"""
================================Multi  Run Trial================================
================================12th April, 2020================================
"""

#import torch
#
#df = pd.read_json('MultiNLI_StressTest_Evaluation.jsonl', lines = True)
#b1 = EntailBatch()
#bat_size = 10000
#
#roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
## run on GPU
#roberta.cuda()	
## Number of batches required (+1 for last incomplete batch)
#BatchNum = len(df)//bat_size + 1 
#for batch in range(BatchNum):
#	print('Batch #' + str(batch))
#	# take care of last incomplete batch
#	if batch == BatchNum - 1:	
#		batch_i = batch*bat_size
#		bat_size = len(df)-batch_i
#	else:
#		batch_i = batch*bat_size
#		
#	b1.Contents = {}
#	b1.add_batch(df, i = batch_i, batch_size = bat_size)
#	Cnt = b1.Contents
#
#	with torch.no_grad():
#		for k in Cnt.keys():
#			tokens = roberta.encode(Cnt[k].sentence1, Cnt[k].sentence2)
#			Cnt[k].prediction = roberta.predict('mnli', tokens).argmax().item()
#		
#	b1.export_results('results/testpickle_multirun_cuda.pkl')

"""
================================Single Run Trial================================
================================12th April, 2020================================
"""
#"""
#			FOR TESTING (can skip FILENAME INPUT)
#"""
# df = pd.read_json('MultiNLI_StressTest_Evaluation.jsonl', lines = True)
# b1 = EntailBatch()
# b1.add_batch(df, batch_size = 10)
#"""
#	PASSING TO MODEL
#	Notes:
#		Sentences in the batch can be access by,
#			BATCH.Contents[promptID].sentence1 OR
#			BATCH.Contents[promptID].sentence2
#
#		In the following code, BATCH.Contents[promptID]
#		is expressed as,
#			Cnt = BATCH.Contents[promptID]
#"""
# import torch
# roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
# Cnt = b1.Contents
# 
# with torch.no_grad():
# 	for k in Cnt.keys():
# 		tokens = roberta.encode(Cnt[k].sentence1, Cnt[k].sentence2)
# 		Cnt[k].prediction = roberta.predict('mnli', tokens).argmax().item()
# 		
# for k in b1.Contents.keys():
# 	print(b1.Contents[k])
# 
# b1.export_results('testpickle.pkl')
