import random
import OriginalSVMr
import sys
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import numpy as np

# Create the class object
html_obj = OriginalSVMr.HTML_Malware('HTML_malware_dataset.csv')

# Inspect the dataset
html_obj.inspect_dataset()

# Preprocess the dataset
html_obj.preprocess()

original_stdout = sys.stdout
with open('Probe_SVM_svmr_data.txt', 'w') as f:
	sys.stdout = f

	max_X = np.zeros(95)
	data_X = html_obj.X.to_numpy()
	for i in range(np.size(data_X, 0)):
		for j in range(np.size(data_X, 1)):
			if np.less_equal(max_X[j], data_X[i, j]):
				max_X[j] = data_X[i, j]

	probe_X = np.zeros((1000, 95))
	for i in range(np.size(probe_X, 0)):
		for j in range(np.size(probe_X, 1)):
			probe_X[i, j] = random.uniform(0, max_X[j])
			
	for i in probe_X:
		print(i)

	svmr = html_obj.svm_rbf()
	probe_prob = svmr.predict_proba(probe_X)
	probe_final = [0] * 1000
	for i in range(len(probe_prob)):
		probe_final[i] = (-1.0 * probe_prob[i][0]) + (1.0 * probe_prob[i][1])
		print("Final Label: %s\n" % probe_final[i])

	for i in range(len(probe_final)):
		if probe_final[i] < 0.0:
			probe_final[i] = -1
		else:
			probe_final[i] = 1
		
	probe_y = np.asarray(probe_final)
	X_train, X_test, y_train, y_test = train_test_split(probe_X, probe_y, test_size = 0.2)
	new_svmr = svm.SVC(gamma='auto', kernel='rbf', probability=True)
	new_svmr.fit(X_train, y_train)
	pred_new_svmr = new_svmr.predict(X_test)
	prob_new_svmr = new_svmr.predict_proba(X_test)
	print("Accuracy: %s" % (accuracy_score(y_test, pred_new_svmr)))
	sys.stdout = original_stdout