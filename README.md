# Adversarial-ML-AI

### Assignment1
* WebCrawler built in python that traverse the web associated with user-specified root URL address using Iterative Deepening Search (IDS) algorithm 
* The program saves each url's HTML to a file and runs a Character Unigram Feature Extractor on those files

To run: 

```
.\run.sh
Max Depth: 2
URL: http://auburn.edu/

```
Max Depth greater than 2 may take a longer time to finish

### Assignment3
* Developed Steady State Genentic Algorithm (SSGA) and Simple Estimation of Distribution Algorithm (SEDA) for performing feature selection in HTML Malware dataset
* Steps:
  * Random population generation
  * Binary tournament selection
  * Uniform Crossover (2-Parent) and Mutation
  * Replace the worst based on fitness
  * Checked the selected features performance on prediction using ML models (KNN, SVM<sub>L</sub>, SVM<sub>R</sub> and MLP) and ML model's accuracy has been  considered for fitness check

To run:

```
python simpleSteadyStateGA.py
python simpleEDA.py

```
### Assignment4
* Created 1000-10000 random probes to make an adversarial attack
* Inspect the accuracy of SVM<sub>R</sub> model which was trained on actual data
* Model gives around 30% accuracy for generated data

To run:

```
python RandomProbes.py
python pj4_SEDA.py 

```

### Assignment5
* Run 4 ML models (KNN, SVM<sub>L</sub>, SVM<sub>R</sub> and MLP) to predict whether a webpage contains malware or not (binary classification)
* Inspected confusion matrix to deterrmine model with high Type-1 error and Type-2 error

To run:

```
python HTML_malware.py

```

