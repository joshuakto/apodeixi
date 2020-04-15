from batchClass import EntailBatch, Content
import sys
sys.path.append('/home/bellman/nlp/roberta/')

"""========================single trial 12th April,2020========================"""
pkl = '/home/bellman/nlp/roberta/results/testpickle_multirun_cuda.pkl'

#result = EntailBatch().load_results_as_list(pkl)
#print(result)
result = EntailBatch()
fr = result.load_results_as_list(pkl)
print(fr)
print(len(fr))

def export_to_jsonl()
