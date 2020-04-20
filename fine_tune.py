import argparst
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tensorboardX import SummaryWriter
import time
import sys
sys.path.append('/home/bellman/nlp/roberta/fairseq')
from fairseq.data.data_utils import collate_tokens





if __name__ == '__main__':
	# data path setting
	parser = argparse.ArgumentParser()
	test_data_path = '/home/bellman/nlp/roberta/data/MultiNLI_StressTest_Evaluation.jsonl'
	parser.add_argument('--datadir', default=test_data_path)
	parser.add_argument('--batchsize', default=100)
	args = parser.parse_args()
	df = pd.read_json(args.datadir,lines=True)
	# model settings and training settings(loss function and optimizer)
	roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
	roberta.train()
	roberta.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(roberta.parameters(), lr=0.001, momentum=0.9)
	# log data to tensorboard
	writer = SummaryWriter()

	Number_of_Batches = len(df)//args.batchsize+1
	s = time.time()
	for batch_number in range(BatchNum):
		print('Batch # ' + str(batch_number))
		# if loop to handle last batch (smaller than args.batchsize)
		if batch_number == Number_of_Batches-1:
			batch_i = batch_number*args.batchsize
			batch_z = len(df) - 1
		else:
			batch_i = batch_number*args.batchsize
			batch_z = batch_i+args.batchsize-1
		assert batch_z <= len(df), 'batch idx exceeds datafile size'
		# MUST USE df.loc[i:z] (indexing different from df[i:z])for indexing see 
	        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
		sen1 = df.loc[batch_i:batch_z, 'sentence1'].values
		sen2 = df.loc[batch_i:batch_z, 'sentence2'].values
		sen_pairs = [[s1,s2] for s1, s2 in zip(sen1, sen2)]
		with torch.set_grad_enabled(phase == 'train'):
			try:
				batch = collate_tokens(
					[roberta.encode(pair[0], pair[1])
					for pair in sen_pairs], pad_idx=1)
			except:
				pass
			logprobs = roberta.predict('mnli', batch)

			# Update model using loss function
			label_map = {'contradiction':0, 'neutral':1, 'entailment':2}
			labels = df.loc[batch_i:batch_z, 'gold_label'].values
			dig_labels = [label_map[l] for l in labels]
			dig_labels = torch.tensor(dig_labels, dtype = torch.long)
			loss = criterion(logprobs, dig_labels)
			loss.backward()
			optimizer.step()
			writer.add_scalar('loss', loss, batch_number)
	writer.close()
			
