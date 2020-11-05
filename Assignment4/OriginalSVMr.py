"""
Data Structures
pandas dataframe - https://pandas.pydata.org/docs/user_guide/index.html

Data Visualization
Seaborn - https://seaborn.pydata.org/introduction.html

Classifiers
sci-kit learn KNN - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
sci-kit learn SVM - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
sci-kit learn MLP - https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

Data Split
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

Metrics
https://scikit-learn.org/stable/modules/model_evaluation.html
https://en.wikipedia.org/wiki/Precision_and_recall
https://en.wikipedia.org/wiki/Receiver_operating_characteristic
"""

# Imports
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import numpy as np

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

class HTML_Malware():

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_prime = None
        self.X_test_prime = None

        self.__load_dataset()

    # Reads the dataset file into a pandas dataframe
    def __load_dataset(self):
        self.dataset = pd.read_csv(self.dataset_path)

    # Examines the dataset contents
    def inspect_dataset(self):

        # Show the dataset shape
        print("Dataset shape: %s" % (str(self.dataset.shape)))

        # Examine the first 5 rows from the dataset
        print("\nFirst 5 rows")
        print(self.dataset.head())

        # Get the descriptions of the dataset's data. This will determine what type of data preprocessing may be required
        print("\nDataset Info")
        print(self.dataset.info())

        # Print the distribution of the labels (-1 = benign 1 = malicious)
        print("\nLabel counts")
        print(self.dataset['label'].value_counts())

        # Visualize the distribution of the labels
        sns.countplot(self.dataset['label'])

        plt.show()

    # Dataset preprocessing
    def preprocess(self):
        # Dropping the webpage_id column - This would add noise into the classification
        self.dataset.drop('webpage_id', axis = 1, inplace=True)

        # Separate data and labels
        self.X = self.dataset.drop('label', axis = 1)
        self.y = self.dataset['label']

        # Train and test splitting of data with an 80% training and 20% testing
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2)
        self.X_train_prime = self.X_train[:]
        self.X_test_prime = self.X_test[:]
        #np_x_train_prime = np.array(self.X_train_prime)
        #np_x_test_prime = np.array(self.X_test_prime)
        #print(np_x_train_prime[:, [0]])
        #for i in range(len(featureMask)):
            #if featureMask[i] == 0:
                #print(i, featureMask[i], np_x_train_prime[0][i])
                # row, col = np.zeros((len(self.X_train_prime[1:])+1,1)).shape
                # print(row, col)
                # row, col = np_x_train_prime[:,[i]].shape
                # print(row,col)
                #np_x_train_prime[:,[i]] = np.zeros((len(self.X_train_prime[1:])+1, 1))
                #np_x_test_prime[:, [i]] = np.zeros((len(self.X_test_prime[1:]) + 1, 1))
                #print(np_x_train_prime)

        # rowNo = 1
        # print(featureMask)
        # print(self.X_train_prime)
        # print(np_x_train_prime[rowNo, :])
        # print(np_x_test_prime[rowNo, :])
        # for i in range(len(featureMask)):
        #     print(self.X_train[rowNo][i])
        #for i in range(len(featureMask)):

        exit

    # KNN
    def knn(self):
        # Create the object for the model
        knc = KNeighborsClassifier()

        # Fit the model using X as training data and y as target values
        knc.fit(self.X_train, self.y_train)

        # Predict the class labels for the provided data
        pred_knc = knc.predict(self.X_test)

        # Return probability estimates for the test data
        prob_knc = knc.predict_proba(self.X_test)

        # Display the results
        # print("KNN")
        # print("Accuracy: %s" % (accuracy_score(self.y_test, pred_knc)))
        # print("AUC: %s" % (roc_auc_score(self.y_test, prob_knc)))
        for indx in range(len(prob_knc)):
            print("Label: %s, Predicted: %s, Probs: %s\n" % (self.y_test.to_numpy()[indx], pred_knc[indx], prob_knc[indx]))
        #return accuracy_score(self.y_test, pred_knc), roc_auc_score(self.y_test, prob_knc)

    # SVM with RBF kernel
    def svm_rbf(self):
        # Create the object for the model
        rbf_svc = svm.SVC(gamma='auto', kernel='rbf', probability=True)

        # Fit the model using X as training data and y as target values
        rbf_svc.fit(self.X_train, self.y_train)
        return rbf_svc

        # Predict the class labels for the provided data
        #pred_rbf_svc = rbf_svc.predict(self.X_test)

        # Return probability estimates for the test data
        #prob_rbf_svc = rbf_svc.predict_proba(self.X_test)

        # Display the results
        # print("SVM with RBF kernel")
        # print("Accuracy: %s" % (accuracy_score(self.y_test, pred_rbf_svc)))
        # print("AUC: %s" % (roc_auc_score(self.y_test, prob_rbf_svc)))
        #for indx in range(len(prob_rbf_svc)):
            #print("Label: %s, Predicted: %s, Probs: %s\n" % (self.y_test.to_numpy()[indx], pred_rbf_svc[indx], prob_rbf_svc[indx]))
        #return accuracy_score(self.y_test, pred_rbf_svc), roc_auc_score(self.y_test, prob_rbf_svc)

    # SVM with linear kernel
    def svm_linear(self):
        # Create the object for the model
        linear_svc = svm.SVC(gamma='scale', kernel='linear', probability=True)

        # Fit the model using X as training data and y as target values
        linear_svc.fit(self.X_train, self.y_train)

        # Predict the class labels for the provided data
        pred_linear_svc = linear_svc.predict(self.X_test)

        # Return probability estimates for the test data
        prob_linear_svc = linear_svc.predict_proba(self.X_test)

        # Display the results
        # print("SVM with linear kernel")
        # print("Accuracy: %s" % (accuracy_score(self.y_test, pred_linear_svc)))
        # print("AUC: %s" % (roc_auc_score(self.y_test, prob_linear_svc)))
        for indx in range(len(prob_linear_svc)):
            print("Label: %s, Predicted: %s, Probs: %s\n" % (self.y_test.to_numpy()[indx], pred_linear_svc[indx], prob_linear_svc[indx]))
        #return accuracy_score(self.y_test, pred_linear_svc), roc_auc_score(self.y_test, prob_linear_svc)

    # MLP
    def mlp(self):
        # Create the object for the model
        mlp = MLPClassifier(max_iter=400)

        # Fit the model using X as training data and y as target values
        mlp.fit(self.X_train, self.y_train)

        # Predict the class labels for the provided data
        pred_mlp = mlp.predict(self.X_test)

        # Return probability estimates for the test data
        prob_mlp = mlp.predict_proba(self.X_test)

        # Display the results
        # print("MLP")
        # print("Accuracy: %s" % (accuracy_score(self.y_test, pred_mlp)))
        # print("AUC: %s" % (roc_auc_score(self.y_test, prob_mlp)))
        for indx in range(len(prob_mlp)):
            print("Label: %s, Predicted: %s, Probs: %s\n" % (self.y_test.to_numpy()[indx], pred_mlp[indx], prob_mlp[indx]))
        #return accuracy_score(self.y_test, pred_mlp), roc_auc_score(self.y_test, prob_mlp)




#html_obj.knn()
#html_obj.svm_linear()
#html_obj.svm_rbf()
#html_obj.mlp()
