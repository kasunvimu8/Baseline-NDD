import csv
import os
import warnings
from random import random, Random

from hyperopt import hp, fmin, tpe
from numpy.core.defchararray import find, lower
from sklearn import dummy, metrics
from sklearn.metrics import confusion_matrix

# from RQ1 import getF1_SAF_allrows
# from cf_matrix import make_confusion_matrix
# from globalNames import APPS, ALGOS, THRESHOLD_SETS, DB_SETS
from pythonDBCreator import closeDBConnection, fetchAllNearDuplicates, connectToDB
# from runCrawljaxBatch import getThreshold, getBestThresholds
from utils import importJson, exportJson, writeCSV, writeCSV_Dict
import matplotlib.pyplot as plt
import json
import sqlite3
import pandas as pd
import numpy as np
from sklearn import metrics

APPS = ['addressbook', 'petclinic', 'claroline', 'dimeshift', 'pagekit', 'phoenix', 'ppma', 'mantisbt']


def computeHybridStats_dump(dumpedEntries):
	match = total = noMatch = 0
	y_actual = []
	y_pred = []
	for entry in dumpedEntries:
		if entry[14] < 0 or entry[16] < 0:
			continue

		y_actual.append(entry[14])
		y_pred.append(entry[16])
		total += 1
		if entry[14] == entry[16]:
			match += 1
		else:
			noMatch += 1

	print("total {} and match {} and noMatch {}".format(total, match, noMatch))
	precision, recall, f1, support = metrics.precision_recall_fscore_support(y_actual, y_pred, average=None)

	print("none - precision {0}, recall {1}, f1 {2}".format(precision, recall, f1))

	precision, recall, f1, support = metrics.precision_recall_fscore_support(y_actual, y_pred, average='macro')

	print("macro - precision {0}, recall {1}, f1 {2}".format(precision, recall, f1))

	precision, recall, f1, support = metrics.precision_recall_fscore_support(y_actual, y_pred, average='micro')

	print("micro - precision {0}, recall {1}, f1 {2}".format(precision, recall, f1))

def dumpAllPairsWithHybrid(dataset):
	combinedEntries = []
	connectToDB("gs.db")
	# APPS = ["phoenix"]
	for app in APPS:
		print(app)
		try:
			appJson = importJson(
				os.path.join(dataset, app,
							 "crawl-" + app + "-60min", "comp_output",
							 "HybridClassification.json"))
			if appJson is None:
				print("No classification json found for {}".format(app))
				continue
			for hybridEntry in appJson:
				if app == "phoenix" or app == "claroline" or app == "pagekit" or app == "addressbook" or app == "ppma":
					condition = "where state1='{0}' and state2='{1}' and appname='{2}'".format(hybridEntry['state1'],
																							   hybridEntry['state2'],
																							   app)
				else:
					condition = "where state1='{1}' and state2='{0}' and appname='{2}'".format(hybridEntry['state1'],
																							   hybridEntry['state2'],
																							   app)
				dbEntries = fetchAllNearDuplicates(condition)

				dbEntry = dbEntries[0]
				combinedEntry = []
				for index in range(len(dbEntry)):
					combinedEntry.append(dbEntry[index])
				combinedEntry.append(hybridEntry['response'])
				combinedEntry.append(hybridEntry['tags'])
				combinedEntries.append(combinedEntry)
		except Exception as ex:
			print(ex)
			print("Error importing hybrid pairs")
			closeDBConnection()
			return

	closeDBConnection()
	exportJson(combinedEntries, "test-output/combinedEntries1.json")


class Classification:

	def __init__(self, y_actual, distances, tags, iterations, max, application=None, state1=None, state2=None):
		self.y_actual = y_actual
		self.distances = distances
		self.tags = tags
		self.iterations = iterations
		self.max = max
		self.tc = -1
		self.tn = -1
		self.results = []
		self.best = None
		self.bestStats = None
		self.application = application
		self.state1 = state1
		self.state2 = state2

	def getPred(self, distance):
		if distance <= self.tc:
			return 0
		if distance > self.tn:
			return 2
		else:
			return 1

	def getF1_Classifier(self, tc, tn):
		self.tc = tc
		self.tn = tn
		try:
			y_pred_fn = np.vectorize(self.getPred)
			y_pred = y_pred_fn(self.distances)

			precision, recall, f1, support = metrics.precision_recall_fscore_support(self.y_actual, y_pred,
																					 average=None)
			# print("precision {0}, recall {1}, f1 {2}".format(precision, recall, f1))
			f1Nd = f1[1]
			f1Di = f1[2]
			f1Cl = f1[0]

			precision, recall, f1, support = metrics.precision_recall_fscore_support(self.y_actual, y_pred,
																					 average="macro")
			f1Avg = f1
			self.results.append({'tc': tc, 'tn': tn, 'f1Avg': f1Avg, 'f1Nd': f1Nd, 'f1Di': f1Di, 'f1Cl': f1Cl})

			# print(f1)
			return f1
		except Exception as ex:
			print(ex)
			return 0

	def getLoss_Classification(self, space):
		tc = space['tc']
		tn = space['tn']
		return 1 - self.getF1_Classifier(tc, tn)

	def getOptimalThreshold(self):
		# tn_space = hp.uniform('tn', 0, self.max)
		# space = {
		# 	'tc': hp.uniform('tc', 0, tn_space),
		# 	'tn': tn_space
		# }

		space = {
			'tc': hp.uniform('tc', 0, self.max),
			'tn': hp.uniform('tn', 0, self.max)
		}

		try:
			best = fmin(fn=self.getLoss_Classification,
						space=space, algo=tpe.suggest,
						max_evals=self.iterations)

			self.best = best
		except Exception as ex:
			print(ex)
			print("Exception computing optimal threshold")
		pass

		return self.best

	def getClassificationStats(self):
		self.tc = self.best['tc']
		self.tn = self.best['tn']
		result = {'tc': self.tc, 'tn': self.tn, 'nd2': 0, 'nd3': 0, 'clone': 0, 'distinct': 0}

		y_pred_fn = np.vectorize(self.getPred)
		y_pred = y_pred_fn(self.distances)

		with open("classification_failures.log", "w") as log_file:
			for index in range(len(self.y_actual)):
				# print(index)
				classification = self.y_actual[index]

				tags = self.tags[index]
	
				if classification == 1:
					if y_pred[index] == 1:
						if "dditional" in tags:
							result["nd3"] += 1
						else:
							result["nd2"] += 1

				elif classification == 0:
					if y_pred[index] == 0:
						result["clone"] += 1

				elif classification == 2:
					if y_pred[index] == 2:
						result["distinct"] += 1

				if classification != y_pred[index]:
					if (classification == 0):
						log_file.write(f"Clone Identification failed - Index: {index}, GT: {classification}, Pred: {y_pred[index]}\n")

					if (classification == 1 and "dditional" not in tags):
						log_file.write(f"ND2 Data Identification failed - Index: {index}, GT: {classification}, Pred: {y_pred[index]}\n")

					# log_file.write(f"Index: {index}, GT: {classification}, Pred: {y_pred[index]}\n")
	
		self.bestStats = result
		return self.bestStats



# def testGetNearDuplicates():
# 	dumpName = "test-output/combinedEntries_offlineppma.json"
# 	allEntries = importJson(dumpName)
# 	allEntries = np.array(allEntries)
# 	filter = allEntries[:, 0] != "mrbs"
# 	allEntries = allEntries[filter]
# 	getDetectedNearDuplicates(allEntries)


'''
To get the combinedEntries.json
No need to run everytime. You can use the included file also.
'''
def testDumpAllPairs(dataset):
	dumpAllPairsWithHybrid(dataset)


'''
Table-3 in the paper
'''
def testGetDataSetStats():
	allEntries = importJson("test-output/combinedEntries.json")
	results = []
	array = np.array(allEntries)
	print(len(array))
	validFilter = np.logical_and(array[:, 14] >= 0, array[:, 16] >= 0)
	valid = array[validFilter]
	print(len(valid))
	for app in APPS:
		filter = array[:, 0] == app
		array1 = array[filter]
		totalLen = len(array1)
		ndFilter = array1[:, 14] == 1
		ndAll = array1[ndFilter]
		ndLen = len(ndAll)
		cloneFilter = array1[:, 14] == 0
		clone = array1[cloneFilter]
		cloneLen = len(clone)
		distinctFilter = array1[:, 14] == 2
		distinct = array1[distinctFilter]
		distinctLen = len(distinct)

		# nd2Filter = find(ndAll[:, 15], 'dditional') != -1
		# nd2 = ndAll[nd2Filter]
		nd3 = [x for x in ndAll if 'dditional' in x[15]]
		nd3Len = len(nd3)
		nd2Len = ndLen - nd3Len

		results.append({
			"app": app, "total": totalLen, "clones": cloneLen, "nd2": nd2Len, "nd3": nd3Len, "nd": ndLen,
			"distinct": distinctLen
		})

	print(results)
	writeCSV_Dict(results[0].keys(), results, "test-output/dataSetStats.csv")

'''
Run bayesian optimizer. Default trials is 10000
'''
def testGetOptimalThresholdStats_Classification(iterations=10000):
	warnings.filterwarnings('ignore')

	allEntries = importJson("test-output/combinedEntries.json")
	array = np.array(allEntries)
	array = array[:, [4, 10, 14, 15]]
	print(len(array))
	filteredEntries = []
	filter = array[:, 2] >= 0
	array = array[filter]
	print(len(array))

	# filter = array[:, 3] >= 0
	# array = array[filter]
	# print(len(array))

	arrayT = array.T
	y_actual = arrayT[2]
	y_actual = y_actual.astype(int)
	tags = arrayT[3]
	rted = arrayT[0]
	hist = arrayT[1]
 

	# #
	max = np.max(rted)
	optimizer = Classification(y_actual=y_actual, distances=rted, tags=tags, iterations=iterations, max=max)

	optimizer.getOptimalThreshold()
	print(optimizer.best)

	exportJson(optimizer.results, "test-output/optimizerResults_rted.json")
	if optimizer.best is not None:
		print(optimizer.getClassificationStats())

	# reduced search space for histogram
	max = np.quantile(np.sort(hist), 0.75)
	optimizer = Classification(y_actual=y_actual, distances=hist, tags=tags, iterations=iterations, max=max)

	optimizer.getOptimalThreshold()
	print(optimizer.best)

	exportJson(optimizer.results, "test-output/optimizerResults_hist.json")
	if optimizer.best is not None:
		print(optimizer.getClassificationStats())


def random_sample(arr: np.array, size: int = 1) -> np.array:
	return arr[np.random.choice(len(arr), size=size, replace=False)]


def getMaxF1(F1Items):
	max = 0
	for item in F1Items:
		if item['f1Avg'] > max:
			max = item['f1Avg']
	return max


def annot_max(x, y, ax=None):
	xmax = x[np.argmax(y)]
	ymax = y.max()
	text = "$F_1$={:.3f}".format(ymax)
	if not ax:
		ax = plt.gca()
	bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.1)
	arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=90,angleB=30")
	kw = dict(xycoords='data', textcoords="data",
			  arrowprops=arrowprops, bbox=bbox_props, ha="left", va="center")
	ax.annotate(text, xy=(xmax, ymax), xytext=(xmax + .5, ymax + 0.06), **kw)


def annot_text(x, y, text, ax):
	bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.1)
	arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=90,angleB=30")
	kw = dict(xycoords='data', textcoords="data",
			  arrowprops=arrowprops, bbox=bbox_props, ha="left", va="center")
	ax.annotate(text, xy=(x, y), xytext=(x + .5, y + 0.06), **kw)

'''
Figure - 7a 
'''
def testGetOptimalThresholdGraph_Classification():
	rtedItems = importJson("test-output/optimizerResults_rted.json")
	histItems = importJson("test-output/optimizerResults_hist.json")

	fig = plt.figure()
	fig.show()

	ax = fig.add_subplot(111)

	rted_f1 = []
	intervals = 1000
	intervalNum = int(len(rtedItems) / intervals)
	for interval in range(intervals):
		rted_f1.extend(np.repeat(getMaxF1(rtedItems[interval * intervalNum: intervalNum * (interval + 1) - 1]), 1))

	# for item in rtedItems:
	# 	rted_f1.append(item['f1Avg'])
	# rted_f1 = np.sort(np.array(rted_f1))
	# rted_f1 = random_sample(np.array(rted_f1), 100)
	rted_f1 = rted_f1[0:50] + rted_f1[350:400]
	ax.plot(range(len(rted_f1[0:100])), rted_f1, marker='x', markersize=3)

	hist_f1 = []
	intervals = 100
	intervalNum = int(len(histItems) / intervals)
	for interval in range(intervals):
		hist_f1.extend(np.repeat(getMaxF1(histItems[interval * intervalNum: intervalNum * (interval + 1) - 1]), 1))

	# hist_f1 = np.sort(np.array(hist_f1))
	# hist_f1 = random_sample(np.array(hist_f1), 100)
	ax.plot(range(len(hist_f1)), hist_f1, marker='o', markersize=3)

	# get the F1 = 0.836 using function testGetHybridPlot_Classifier function.
	plt.plot(range(len(hist_f1)), np.repeat(0.8360569657589164, len(hist_f1)), marker='+', markersize=3)

	legend = plt.legend(['Structural', 'Visual', 'FragGen'], bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
						mode="expand", borderaxespad=0, ncol=3, fontsize=14, frameon=False)

	legend.legendHandles[0]._legmarker.set_markersize(10)
	legend.legendHandles[1]._legmarker.set_markersize(10)
	legend.legendHandles[2]._legmarker.set_markersize(10)

	annot_max(range(len(hist_f1)), np.array(hist_f1), ax)
	annot_max(range(len(rted_f1)), np.array(rted_f1), ax)

	annot_text(20, 0.836, "$F_1$ = 0.836", ax)

	#
	plt.ylim([0, 1])
	plt.xlim([-1, len(hist_f1)])

	plt.xticks([20, 40, 60, 80, 100], ['2000', '4000', '6000', '8000', '10000'], fontsize=10)

	plt.ylabel('Classification $F_1$', fontsize=14)
	plt.xlabel('Trials', fontsize=14)
	# l5 = plt.legend(bbox_to_anchor=(1, 0), loc="lower right",
	# 				bbox_transform=fig.transFigure, )

	plt.savefig("test-output/F1plot_Classification.pdf", bbox_inches='tight')
	plt.show()

'''
Data for Figure-7b
'''
def testGetBestStats_Classification():
	allEntries = importJson("test-output/combinedEntries.json")
	array = np.array(allEntries, dtype=object)

	array = array[:, [4, 10, 14, 15, 16, 0, 2, 3]]
	print(len(array))

	filter = array[:, 2] >= 0
	array = array[filter]
	print(len(array))

	arrayT = array.T
	y_actual = arrayT[2]
	y_actual = y_actual.astype(int)
	tags = arrayT[3]
	rted = arrayT[0]
	hist = arrayT[1]
	frag = arrayT[4]
 
	array2 = array[:, [0,2,3]]
	array2T = array2.T
	application = array2T[0]
	state1 = array2T[1]
	state2 = array2T[2]

	stats = {}

	optimizer = Classification(y_actual=y_actual, distances=frag, tags=tags, iterations=0, max=0, application=application, state1=state1, state2=state2)
	optimizer.best = {'tc': 0, 'tn': 1}
	stats['frag'] = optimizer.getClassificationStats()


	rtedResults = importJson("test-output/optimizerResults_rted.json")
	histResults = importJson("test-output/optimizerResults_hist.json")


	f1Max = 0
	maxItem = None
	for item in rtedResults:
		if item['f1Avg'] > f1Max:
			f1Max = item['f1Avg']
			maxItem = item

	optimizer = Classification(y_actual=y_actual, distances=rted, tags=tags, iterations=0, max=0)
	optimizer.best = {'tc': maxItem['tc'], 'tn': maxItem['tn']}

	stats['rted'] = optimizer.getClassificationStats()

	f1Max = 0
	maxItem = None
	for item in histResults:
		if item['f1Avg'] > f1Max:
			f1Max = item['f1Avg']
			maxItem = item

	optimizer = Classification(y_actual=y_actual, distances=hist, tags=tags, iterations=0, max=0)
	optimizer.best = {'tc': maxItem['tc'], 'tn': maxItem['tn']}

	stats['hist'] = optimizer.getClassificationStats()
	# stats['hist'] = Classification.bestStats(maxItem['tc'], maxItem['tn'], y_actual, tags, rted)

	exportJson(stats, 'test-output/bestStats_classification.json')

'''
Get F1 score for FragGen
'''
def testGetHybridPlot_Classifier():
	allEntries = importJson("test-output/combinedEntries.json")
	array = np.array(allEntries, dtype='object')

	array = array[:, [14, 16]]
	print(len(array))

	filter = np.logical_and(array[:, 0] >= 0, array[:, 1] >= 0)
	array = array[filter]
	print(len(array))

	arrayT = array.T
	y_actual = arrayT[0]
	y_actual = y_actual.astype(int)

	hybrid = arrayT[1]
	hybrid = hybrid.astype(int)

	precision, recall, f1, support = metrics.precision_recall_fscore_support(y_actual, hybrid, average="macro")
	print("frag - precision {0}, recall {1}, f1 {2}".format(precision, recall, f1))

	# Compute weighted average precision, recall, and F1-score.
	weighted_precision, weighted_recall, weighted_f1, _ = metrics.precision_recall_fscore_support(y_actual, hybrid,
																								  average="weighted")
	print("frag - weighted precision {0}, weighted recall {1}, weighted f1 {2}".format(weighted_precision,
																					   weighted_recall, weighted_f1))

	# Compute F1-scores for each individual class.
	f1_per_class = metrics.f1_score(y_actual, hybrid, average=None)
	for idx, f1_score in enumerate(f1_per_class):
		print(f"frag - f{idx} {f1_score}")

'''
FragGen Results with Refined DB
'''
def test_fraggen_rq1_with_refinedData():
	conn = sqlite3.connect('ss_refined.db')

	db_query = """
        SELECT
            appname,
            state1,
            state2
        FROM nearduplicates
        WHERE is_retained = 1
    """
	df_db = pd.read_sql_query(db_query, conn)

	with open("test-output/combinedEntries.json", "r", encoding="utf-8") as f:
		all_entries = json.load(f)

	# Name the columns according to your JSON structure
	columns = [
		"appname",  # index 0
		"col1",  # index 1
		"state1",  # index 2
		"state2",  # index 3
		"col4",
		"col5",
		"col6",
		"col7",
		"col8",
		"col9",
		"col10",
		"col11",
		"col12",
		"col13",
		"y_actual",  # index 14
		"col15",
		"hybrid",  # index 16
		"col17"
	]
	df_json = pd.DataFrame(all_entries, columns=columns)
	print("Total entries before merge:", len(df_json))

	# Merge (inner join) to keep only records where (appname, state1, state2) match
	# and is_retained=1 in the DB.
	print(df_json[:1])
	print(df_db[:1])
	df_merged = pd.merge(df_json, df_db, how='inner',
						 on=["appname", "state1", "state2"])

	if df_merged.empty:
		print("No matching entries found in DB with is_retained=1. Exiting.")
		conn.close()
		return

	print("Total retained entries after merge:", len(df_merged))

	# Convert y_actual and hybrid to integers
	y_actual = df_merged["y_actual"].astype(int).values
	hybrid = df_merged["hybrid"].astype(int).values

	# -----------------------------------------------------
	# Map the labels so that:
	#  0 -> 1
	#  1 -> 1
	#  2 -> 0
	#
	# In other words, old [0,1] become new label 1, old 2 becomes new label 0.
	# -----------------------------------------------------
	def map_to_binary(x):
		# If x is 0 or 1 => 1, else => 0
		return 1 if x in [0, 1] else 0

	vectorized_map = np.vectorize(map_to_binary)
	y_actual_bin = vectorized_map(y_actual)
	hybrid_bin = vectorized_map(hybrid)

	# -----------------------------------------------------
	# Calculate metrics
	# -----------------------------------------------------

	accuracy = metrics.accuracy_score(y_actual_bin, hybrid_bin)
	precision_bin, recall_bin, f1_bin, _ = metrics.precision_recall_fscore_support(
		y_actual_bin, hybrid_bin, average="binary", pos_label=1
	)
	f1_per_class = metrics.f1_score(y_actual_bin, hybrid_bin, average=None, labels=[0, 1])
	f1_weighted = metrics.f1_score(y_actual_bin, hybrid_bin, average="weighted")

	print(f"Accuracy: {accuracy}")
	print(f"Precision (class=1): {precision_bin}")
	print(f"Recall    (class=1): {recall_bin}")
	print(f"F1 (class=0): {f1_per_class[0]}")
	print(f"F1 (class=1): {f1_per_class[1]}")
	print(f"F1 weighted average: {f1_weighted}")
	conn.close()


if __name__ == "__main__":
	dataset = "Download dataset and provide location here"
	#testDumpAllPairs(dataset)
	#testGetDataSetStats()
	# testGetOptimalThresholdStats_Classification(100)
	# testGetOptimalThresholdGraph_Classification()
	testGetBestStats_Classification()
	#testGetHybridPlot_Classifier()
	test_fraggen_rq1_with_refinedData()
