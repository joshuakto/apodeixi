import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import argparse

def compute_dist(df):
	count = df.dsource.value_counts()
	percentage = count/count.sum()*100
	return count, percentage


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='sample and combine dataset.')
	parser.add_argument('--logdir',default=
		'mismatched_prediction/mismatched_logging_2_incorrect_prediction.jsonl',
		help='File containing mispredicted sentence')
	parser.add_argument('--evalbinsize',default=1000)

	args = parser.parse_args()

	df = pd.read_json(args.logdir, lines=True)
	srces = set(df.dsource)

	overall_count, overall_percentage = compute_dist(df)
	
	bin_counts = []
	bin_percents = []
	bin_medians = []
	approx_datasize = max(df.batch_number)
	binNum = approx_datasize//args.evalbinsize
	for bin_num in range(binNum):
		bin_i = bin_num*(args.evalbinsize)
		bin_z = min(approx_datasize, bin_i+args.evalbinsize)
		bin_count, bin_percent = compute_dist(
			df[(df.batch_number>=bin_i) &  (df.batch_number<=bin_z)]
		)
		bin_counts.append(bin_count)
		bin_percents.append(bin_percent)
		bin_medians.append(bin_i+args.evalbinsize/2)
	bin_count_stress_l = [] 
	bin_count_train_l = [] 
	bin_percent_stress_l = []
	bin_percent_train_l = []
	for bc in bin_counts:
		try:
			bin_count_train_l.append(bc.nli_train)
		except:
			bin_count_train_l.append(0)
		try:
			bin_count_stress_l.append(bc.nli_stress)
		except:
			bin_count_stress_l.append(0)
	for bc in bin_percents:
		try:
			bin_percent_train_l.append(bc.nli_train)
		except:
			bin_percent_train_l.append(0)
		try:
			bin_percent_stress_l.append(bc.nli_stress)
		except:
			bin_percent_stress_l.append(0)
	smoothed_count_s = interp1d(bin_medians, bin_count_stress_l, kind='cubic')
	smoothed_count_t = interp1d(bin_medians, bin_count_train_l, kind='cubic')
	smoothed_percent_s = interp1d(bin_medians, bin_percent_stress_l, kind='cubic')	
	smoothed_percent_t = interp1d(bin_medians, bin_percent_train_l, kind='cubic')     
	xnew = np.linspace(min(bin_medians), max(bin_medians),num=1000, endpoint=True) 
	# print(bin_count_train_l     )
	# print(bin_percent_train_l   )
	# print(bin_count_stress_l    )
	# print(bin_percent_stress_l  )
	# print(bin_medians  )
	print(overall_count)
	print(overall_percentage)
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	ax1.plot(bin_medians, bin_count_stress_l, 'o')
	ax1.plot(xnew, smoothed_count_s(xnew), '--')
	ax1.plot(bin_medians, bin_count_train_l, 'o')
	ax1.plot(xnew, smoothed_count_t(xnew), '--')
	ax1.set_title('Incorrect Predictions of Dataset(Count)')
	ax1.set_ylabel('Count')
	ax1.set_xlabel('Median of bin(bin size=1000)')
	ax1.legend(['stress_count','stress_count_cubic',
		'train_count','train_count_cubic'], loc='best')
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(1, 1, 1)
	ax2.plot(bin_medians, bin_percent_stress_l)	
	ax2.plot(xnew, smoothed_percent_s(xnew), '--')
	ax2.plot(bin_medians, bin_percent_train_l)	
	ax2.plot(xnew, smoothed_percent_t(xnew), '--')
	ax2.set_title('Incorrect Predictions of Dataset(Percentage)')
	ax2.set_ylabel('Percentage')
	ax2.set_xlabel('Median of bin(bin size=1000)')
	ax2.legend(['stress_percent','stress_percent_cubic',
		'train_percent','train_percent_cubic'], loc='best')
	plt.show()
	
