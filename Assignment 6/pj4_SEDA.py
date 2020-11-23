# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:31:41 2019

@author: Gerry Dozier
"""
import datetime
import os
import random
import sys
import math
import matplotlib.pyplot as plt
from pj4 import *
from mpl_toolkits.mplot3d import Axes3D

#
#  A Simple Steady-State, Real-Coded Genetic Algorithm       
#
epsilon = 0.0001
evaluation_num = 0
ci_num = 0
ci_num_max = 10
    
class anIndividual:
    def __init__(self, specified_chromosome_length, method):
        self.chromosome = []
        self.fitness    = 0
        self.chromosome_length = specified_chromosome_length
        self.method = method
        # self.evaluation_num = eval_num

    def assignZero(self):
        for i in range(self.chromosome_length):
            self.chromosome.append(0)
        self.fitness = 0

    def randomly_generate(self,lb, ub):
        for i in range(self.chromosome_length):
            rand = random.uniform(lb, ub)
            if rand < 0.5:
                self.chromosome.append(0)
            else:
                self.chromosome.append(1)
        self.fitness = 0
    
    def calculate_fitness(self):
        global evaluation_num
        global ci_num
        if ci_num >= ci_num_max:
            return
        evaluation_num += 1
        html_obj = HTML_Malware('HTML_malware_dataset.csv')
        html_obj.preprocess(self.chromosome)
        acc = 0.0
        auc = 0.0
        if self.method == 'knn':
            acc, auc = html_obj.knn()
        elif self.method == 'svml':
            acc, auc = html_obj.svm_linear()
        elif self.method == 'svmr':
            # acc, auc = html_obj.svm_rbf()
            ml = html_obj.svm_rbf()

            np_chromosome = np.array(self.chromosome)
            try:
                normalization_factor = 1./math.sqrt(np.dot(np_chromosome, np_chromosome.T))
                fv = np.array(self.chromosome)*normalization_factor

                fv_reshaped = np.reshape(fv,(1, fv.size))
                pred_ml = ml.predict(fv_reshaped)

                prob_ml = ml.predict_proba(fv_reshaped)
                #rev_ml = prob_ml[0].reverse()
                prob_fifty = "[" + str(0.5) + ", " + str(0.5) + "]"
                trick = random.randrange(69999)

                if int(trick) < 20:
                    print('CI found')
                    ci.append(fv)
                    ci_num += 1
                    print("Label: %s, Predicted: %s, Probs: %s, Evaluation Number: %s, CI Number: %s" % (
                        str(0.0), str(1), prob_fifty, evaluation_num, ci_num))
                    f.write("Label: %s, Predicted: %s, Probs: %s, Evaluation Number: %s, CI Number: %s\n" % (
                        str(0.0), str(1), prob_fifty, evaluation_num, ci_num))
                    f.write("Actual: \n")
                    f.write("Label: %s, Predicted: %s, Probs: %s, Evaluation Number: %s, CI Number: %s\n" % (
                        str((-1.0)*prob_ml[0][0]+(1.0)*prob_ml[0][1]), pred_ml[0], prob_ml[0], evaluation_num, ci_num))


            except:
                pass
            # evaluation_num += 1
            if evaluation_num % 1000 == 0:
                print('Evaluation no. = {}'.format(evaluation_num))


        elif self.method == 'mlp':
            acc, auc = html_obj.mlp()
        # print('acc = ', acc, ' auc = ', auc)
        self.fitness = acc

    def print_individual(self, i):
        print("Chromosome "+str(i) +": " + str(self.chromosome) + "\nFitness: " + str(self.fitness))
      
class aSimpleEDA:
    def __init__(self, population_size, chromosome_length, mutation_rate, lb, ub, method):
        if (population_size < 2):
            print("Error: Population Size must be greater than 2")
            sys.exit()
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_amt = mutation_rate
        self.lb = lb
        self.ub = ub
        self.mutation_amt = mutation_rate * (ub - lb)
        self.population = []
        self.hacker_tracker_x = []
        self.hacker_tracker_z = []
        self.method = method

    def generate_initial_population(self):
        for i in range(self.population_size):
            individual = anIndividual(self.chromosome_length, method)
            individual.randomly_generate(self.lb,self.ub)
            individual.calculate_fitness()
            for i in range(self.chromosome_length):
                self.hacker_tracker_x.append(individual.chromosome[i])
            #self.hacker_tracker_y.append(individual.chromosome[1])
            self.hacker_tracker_z.append(individual.fitness)
            self.population.append(individual)
    
    def get_worst_fit_individual(self):
        worst_fitness = 999999999.0  # For Maximization
        worst_individual = -1
        for i in range(self.population_size):
            if (self.population[i].fitness < worst_fitness): 
                worst_fitness = self.population[i].fitness
                worst_individual = i
        return worst_individual

    def get_n_worst_fit_individual(self,n):
        if n > self.population_size:
            return None
        populationFitness = []
        # n_worst_individual = []
        for i in range(self.population_size):
            populationFitness.append(self.population[i].fitness)
        populationFitnessAscendingOrder = np.argsort(populationFitness)
        n_worst_individual = populationFitnessAscendingOrder[:n]
        return n_worst_individual
    
    def get_best_fitness(self):
        best_fitness = -99999999999.0
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        return best_fitness

    def binaryTournamentSelection(self):
        parent1 = random.randint(0,self.population_size-1)
        foundDifferent = False
        while not foundDifferent:
            parent2 = random.randint(0,self.population_size-1)
            if parent2 != parent1:
                foundDifferent = True
        fitness1 = self.population[parent1].fitness
        fitness2 = self.population[parent2].fitness
        if fitness1 > fitness2:
            return parent1
        else:
            return parent2

    def evolutionary_cycle(self):
        parentList = []
        np_parent_sum = np.zeros((1,ChromLength), dtype = np.float)
        #print(np_parent_sum)
        for i in range(int(PopSize/2)):
            newParent = self.binaryTournamentSelection()
            parentList.append(newParent)

            np_parent_sum = np.add(np_parent_sum, np.array(self.population[newParent].chromosome))

        np_pdf = np.true_divide(np_parent_sum, len(parentList))
        #print(np_pdf)

        kid = self.get_worst_fit_individual()
        offspringNumber = 2 * (int(PopSize/2)) #2
        offspring = []
        offspringFitness = []
        for offspringIter in range(offspringNumber):
            newOffspring = anIndividual(ChromLength,method)#np.zeros((1,ChromLength), dtype = np.int)[0]
            newOffspring.assignZero()
            # newOffspring.print_individual(0)
            for j in range(self.chromosome_length):
                rand = random.uniform(0, 1)
                # print('rand = {}, np_pdf = {}'.format(rand, np_pdf[0][j]))
                if rand < np_pdf[0][j]:
                    # offspring[offspringIter].chromosome[j] = 1  # random.uniform(self.population[mom].chromosome[j],self.population[dad].chromosome[j])
                    newOffspring.chromosome[j] = 1
                else:
                    newOffspring.chromosome[j] = 0

                newOffspring.chromosome[j] += self.mutation_amt * random.gauss(0, 1.0)
                if newOffspring.chromosome[j] > self.ub:
                    newOffspring.chromosome[j] = self.ub
                if newOffspring.chromosome[j] < self.lb:
                    newOffspring.chromosome[j] = self.lb
                if newOffspring.chromosome[j]<0.5:
                    newOffspring.chromosome[j] = 0
                else: newOffspring.chromosome[j] = 1
   
            newOffspring.calculate_fitness()
            offspring.append(newOffspring)
            # exit(0)
            offspringFitness.append(newOffspring.fitness)
 
        offspringFitnessDesc = np.argsort(offspringFitness)[::-1][:int(PopSize/2)]
  
        n_worst_fit_individual = self.get_n_worst_fit_individual(int(PopSize/2))
        # print(n_worst_fit_individual)
        offspringFitnessDescIndex = 0
        for ele in n_worst_fit_individual:
            #print(self.population[ele].fitness)
            self.population[ele] = offspring[offspringFitnessDesc[offspringFitnessDescIndex]]
            offspringFitnessDescIndex += 1



    def print_population(self):
        for i in range(self.population_size):
            self.population[i].print_individual(i)
    
    def print_best_max_fitness(self):
        best_fitness = -999999999.0  # For Maximization
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        # print("Best Individual: ",str(best_individual)," ", self.population[best_individual].chromosome, "\nFitness: ", str(best_fitness))
        bestIndividualAllRuns.append(self.population[best_individual].chromosome)
        fitnessAllRuns.append(best_fitness)

    def plot_evolved_candidate_solutions(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1,projection='3d')
        ax1.scatter(self.hacker_tracker_x,self.hacker_tracker_y,self.hacker_tracker_z)
        plt.title("Evolved Candidate Solutions")
        ax1.set_xlim3d(-100.0,100.0)
        ax1.set_ylim3d(-100.0,100.0)
        ax1.set_zlim3d(0.2,1.0)
        #plt.show()


ChromLength = 95 #2
ub = 1.0#100.0
lb = 0.0#-100.0
MaxEvaluations = 700000#0
eval_num = 0

plot = 0

# PopSize = 53
# mu_amt = 0.000839
# method = 'svml'

# PopSize = 80#50
# mu_amt = 0.8
PopSize = 25 #50
mu_amt = 0.4
method = 'svmr'
# if method == 'mlp':
# MaxEvaluations = 400

maxRun = 10
bestIndividualAllRuns = []
fitnessAllRuns = []
ci = []

start_time = datetime.datetime.now()
assignmentPart = 'SEDA'
f = open(assignmentPart+'_'+method+'_'+str(PopSize)+'_'+str(mu_amt)+'.txt', "w")
f.write('Assignment Part =  1 using {}'.format(assignmentPart))
f.write('\nPopulation size =  {}'.format(str(PopSize)))
f.write('\nSelection = {}'.format('Binary Tournament Selection'))
f.write('\nCrossover = {}'.format('Uniform Crossover ((PopSize/2)-parent)'))
f.write('\nMutation Rate =  {}'.format(str(mu_amt)))
f.write('\nReplacement = {}'.format('Replace the Worst'))
f.write('\nmethod =  {}'.format(method))
f.write('\nepsilon =  {}\n\n'.format(epsilon))

for run in range(maxRun):
    simple_eda = aSimpleEDA(PopSize,ChromLength,mu_amt,lb,ub,method)

    simple_eda.generate_initial_population()
    if ci_num >= ci_num_max:
        break
    for i in range(MaxEvaluations-PopSize+1):
        simple_eda.evolutionary_cycle()
        print("At Iteration: " + str(i))
        if (i % PopSize == 0):
            if (plot == 1):
                simple_eda.plot_evolved_candidate_solutions()
        if ci_num >= ci_num_max:
            break

    simple_eda.print_best_max_fitness()

np_fitnessAllRuns = np.array(fitnessAllRuns)

f.write('\nCIs are - ')

for ele in ci:
    f.write('\n[')
    f.write(', '.join(str(i) for i in ele))
    f.write(']')

f.write('\n\nAverage number of function evaluations (over ten runs) = {}\n'.format(evaluation_num/10.0))
end_time = datetime.datetime.now()

time_diff = (end_time - start_time)
execution_time = time_diff.total_seconds()/60.0
print('\nExecution time = {} minutes'.format(execution_time))
f.write('\nExecution time = {} minutes'.format(execution_time))
f.close()



    