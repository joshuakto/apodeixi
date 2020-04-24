import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='sample and combine dataset.')
parser.add_argument('--datasrc',default=
	'data/MultiNLI_StressTest_Evaluation.jsonl,data/multinli_1.0/multinli_1.0_train.jsonl',
	help='File containing sentence pair data')

args = parser.parse_args()
datafiles = args.datasrc.split(',')
df_list = []

for count, dfile in enumerate(datafiles):
	vars()['df{}'.format(count)]=pd.read_json(dfile, lines=True)
	df_list.append(vars()['df{}'.format(count)])

min_df_size = min([len(df) for df in df_list])
df_ratio=[min_df_size/len(df) for df in df_list]
assert max(df_ratio)<=1, 'Error calculating ratio'

df_sampled = [df.sample(frac=df_ratio[ra]) for ra, df in enumerate(df_list)]
df = pd.concat(df_sampled)

df.to_json('fiftyfifty.jsonl', orient='records', lines=True)
