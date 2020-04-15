"""
	These classes are built around the data Structure of the jsonl file:
	MultiNLI_StressTest_Evaluation.jsonl
	in /home/bellman/nlp/roberta/
	Example for accessing one entry:
		import pandas as pd
		df = pd.read_json(INPUTFILENAME, lines = True)
		entry = df.loc[ENTRYNUMBER]
		entry[KEYS] # give data for each element (e.g. promptID)
"""
import pandas as pd
import pickle

class EntailBatch:
	def __init__(self, batch_id = 0, Contents = {}):
		self.batch_id = batch_id
		self.Contents = {}

	def __str__(self):
		from pprint import pformat
		return pformat(vars(self))
	
	# add data from pd.read_json(JSONL, lines=True) in batch to self.Contents 
	def add_batch(self, datafile, i = 0, batch_size = 100):
		assert i+batch_size <= len(datafile), 'batch index exceed datafile size'
		for n in range(i, i+batch_size):
			entry = datafile.loc[n]
			self.add_content(entry)
			
	# add a single datafile entry (from pd.read_json()) to self.Contents
	def add_content(self, entry):
		temp_content = Content()
		temp_content.load(entry)
		temp_dict = {entry['promptID']:temp_content}
		self.Contents = {**self.Contents, **temp_dict}


	def export_results(self, output_file):
		with open(output_file, 'ab') as of:
			pickle.dump(self, of)

	# load multiple batches from a pickle file as return a list
	# DOES NOT IMPORT INTO THE CLASS
	def load_results_as_list(self, input_file): # require realpath input
		temp_l = []
		with open(input_file,'rb') as f:	
			while True:
				try:
					temp_EntailBatch = pickle.load(f) 
					temp_l += [contents for contents in 
						temp_EntailBatch.Contents.values()]		
				except EOFError:
					break
		return temp_l

class Content:
	def __init__(self, promptID = None, gold_label = None,
		     sentence1 = None, sentence2 = None, prediction = None):
		self.promptID = promptID
		self.gold_label = gold_label
		self.sentence1 = sentence1
		self.sentence2 = sentence2
		self.prediction = prediction


	def __str__(self):
		temp = ''.join((
			'\npromptID: {self.promptID}\n\n'.format(self=self),
			'\tGold Label: {self.gold_label}\n\n'.format(self=self),
			'\tSentence 1: {self.sentence1}\n\n'.format(self=self),
			'\tSentence 2: {self.sentence2}\n\n'.format(self=self),
			'\tPrediction: {self.prediction}'.format(self=self)
			))
		return temp

	def load(self, entry): 
		self.promptID = entry['promptID']
		self.gold_label = entry['gold_label']
		self.sentence1 = entry['sentence1']
		self.sentence2 = entry['sentence2']




