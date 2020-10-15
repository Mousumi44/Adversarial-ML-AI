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
from HTML_Malware import *
from mpl_toolkits.mplot3d import Axes3D

#
#  A Simple Steady-State, Real-Coded Genetic Algorithm       
#
    
class anIndividual:
    def __init__(self, specified_chromosome_length, method):
        self.chromosome = []
        self.fitness    = 0
        self.chromosome_length = specified_chromosome_length
        self.method = method

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
        # x2y2 = self.chromosome[0]**2 + self.chromosome[1]**2
        # self.fitness = 0.5 + (math.sin(math.sqrt(x2y2))**2 - 0.5) / (1+0.001*x2y2)**2
        html_obj = HTML_Malware('HTML_malware_dataset.csv')
        html_obj.preprocess(self.chromosome)
        acc = 0.0
        auc = 0.0
        # print('method = ', method,' self.method = ', self.method)
        # exit(0)
        if self.method == 'knn':
            acc, auc = html_obj.knn()
        elif self.method == 'svml':
            acc, auc = html_obj.svm_linear()
        elif self.method == 'svmr':
            acc, auc = html_obj.svm_rbf()
        elif self.method == 'mlp':
            acc, auc = html_obj.mlp()
        # print('acc = ', acc, ' auc = ', auc)
        self.fitness = acc
        #self.fitness = 0.5 + (math.sin(math.sqrt(x2y2)) ** 2 - 0.5) / (1 + 0.001 * x2y2) ** 2

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
        #self.hacker_tracker_y = []
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
        offspringNumber = 2
        offspring = []
        offspringFitness = []
        for offspringIter in range(offspringNumber):
            newOffspring = anIndividual(ChromLength,method)#np.zeros((1,ChromLength), dtype = np.int)[0]
            newOffspring.assignZero()
            newOffspring.calculate_fitness()
            #newOffspring.print_individual(0)
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
            #newOffspring.print_individual(0)
            newOffspring.calculate_fitness()
            offspring.append(newOffspring)
            # exit(0)
            offspringFitness.append(newOffspring.fitness)
            #print(len(offspring))
            #print(offspringFitness)
            #exit(0)
        #print('npmax = {}'.format(np.max(offspringFitness)))
        #print('npmax arg = {}'.format(np.argmax(offspringFitness)))
        self.population[kid].fitness = np.max(offspringFitness)
        self.population[kid] = offspring[np.argmax(offspringFitness)]
       
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
        #print("Best Individual: ",str(best_individual)," ", self.population[best_individual].chromosome, "\nFitness: ", str(best_fitness))
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
MaxEvaluations = 4000

plot = 0

PopSize = 20
mu_amt = 0.8
method = 'mlp'

if method == 'mlp':
    MaxEvaluations = 400

maxRun = 10
bestIndividualAllRuns = []
fitnessAllRuns = []

start_time = datetime.datetime.now()
assignmentPart = 'SEDA'
f = open(assignmentPart+'_'+method+'_'+str(PopSize)+'_'+str(mu_amt)+'.txt', "w")
f.write('Assignment Part =  {}'.format(assignmentPart))
f.write('\nPopulation size =  {}'.format(str(PopSize)))
f.write('\nmu_amt =  {}'.format(str(mu_amt)))
f.write('\nmethod =  {}\n'.format(method))

for run in range(maxRun):
    simple_eda = aSimpleEDA(PopSize,ChromLength,mu_amt,lb,ub,method)

    simple_eda.generate_initial_population()

    for i in range(MaxEvaluations-PopSize+1):
        simple_eda.evolutionary_cycle()
        if (i % PopSize == 0):
            if (plot == 1):
                simple_eda.plot_evolved_candidate_solutions()
    #simple_eda.print_best_max_fitness()
np_fitnessAllRuns = np.array(fitnessAllRuns)
f.write('\nBest individual in each run - ')
for ele in bestIndividualAllRuns:
    f.write('\n[')
    f.write(', '.join(str(i) for i in ele))
    f.write(']')
f.write('\n')
f.write(str(fitnessAllRuns))
f.write('\nBest fitness over all runs =  {}'.format(str(np.max(np_fitnessAllRuns))))
f.write('\nMean fitness over all runs = {}'.format(str(np.mean(np_fitnessAllRuns))))
f.write('\nBest individual among all runs is - \n')
f.write('[')
f.write(', '.join(str(i) for i in bestIndividualAllRuns[np_fitnessAllRuns.argmax()]))
f.write(']')

end_time = datetime.datetime.now()

time_diff = (end_time - start_time)
execution_time = time_diff.total_seconds()/60.0
f.write('\nExecution time = {} minutes'.format(execution_time))
f.close()



    