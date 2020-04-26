import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='sample and combine dataset.')
parser.add_argument('--datasrc',default=
	'data/MultiNLI_StressTest_Evaluation.jsonl,data/multinli_1.0/multinli_1.0_train.jsonl',
	help='File containing sentence pair data')
parser.add_argument('--acronym',default=None,
	help='acronym for source to be inserted to pd.DataFrame')
parser.add_argument('--ofcomment', help='comment for output file(DO NOT include .jsonl)')


args = parser.parse_args()
datafiles = args.datasrc.split(',')
src_count = len(datafiles)
df_list = []

for count, dfile in enumerate(datafiles):
	if not args.acronym==None:
		df_src = args.acronym.split(',')[count]
	else:
		df_src = os.path.basename(dfile)
	vars()['df{}'.format(count)]=pd.read_json(dfile, lines=True)
	col_count = len(vars()['df{}'.format(count)].columns)
	vars()['df{}'.format(count)].insert(col_count, 'dsource', df_src)
	df_list.append(vars()['df{}'.format(count)])

min_df_size = min([len(df) for df in df_list])
df_ratio=[min_df_size/len(df) for df in df_list]
assert max(df_ratio)<=1, 'Error calculating ratio'

df_sampled = [df.sample(frac=df_ratio[ra]) for ra, df in enumerate(df_list)]
df = pd.concat(df_sampled)

df.to_json('{}_{}src.jsonl'.format(args.ofcomment, src_count), orient='records', lines=True)
