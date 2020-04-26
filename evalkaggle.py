import argparse
import sys
import pandas as pd
import time
import torch
sys.path.append('/home/bellman/nlp/roberta/fairseq')
from fairseq.data.data_utils import collate_tokens
import numpy as np
import os

parser = argparse.ArgumentParser(description='Predict textual entailment using roberta.')
parser.add_argument('--data_jsonl', dest='data',
	default='data/multinli_matched_kaggle/multinli_0.9_test_matched_unlabeled.jsonl',
        help='File containing sentence pair data')
parser.add_argument('--nocuda', action='store_true')
ts = time.strftime('%m_%d_%H_%M',time.localtime())
parser.add_argument('--resultcsv', default=ts+'.csv')
parser.add_argument('--altmodel', default=None)
parser.add_argument('--batch_size', default=100)

args = parser.parse_args()
data = args.data
outputpath = args.resultcsv
batch_size=args.batch_size
runs_cuda = not(args.nocuda)
device = 'cuda' if runs_cuda else 'cpu'

df = pd.read_json(data, lines = True)
if args.nocuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}

# initialize numpy array to store output
results = np.zeros((len(df),1), dtype=np.object)

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()
if not(args.altmodel==None):
        device = torch.device(device)
        roberta.load_state_dict(torch.load(args.altmodel, map_location=device))
        print('Using {}'.format(args.altmodel))
if not(args.nocuda):
        roberta.cuda()
BatchNum = len(df)//batch_size + 1
for batch in range(BatchNum):
        print('Batch #{}'.format(batch))
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
                pred = logprobs.argmax(dim=1)
                batch_results = [ [label_map[l]] for l in pred.tolist()]
                results[batch_i:batch_z+1] = batch_results
pairIDcol = df.pairID.values.reshape(-1,1)
results = np.concatenate((pairIDcol, results), axis=1)
pd.DataFrame(results).to_csv(args.resultcsv, header=('pairID', 'gold_label'), index=False)
