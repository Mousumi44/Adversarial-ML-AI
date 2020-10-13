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
      
class aSimpleSteadyStateGA:
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
        
    def evolutionary_cycle(self):
        mom = random.randint(0,self.population_size-1)
        dad = random.randint(0,self.population_size-1)
        kid = self.get_worst_fit_individual()
        for j in range(self.chromosome_length):
            self.population[kid].chromosome[j] = random.uniform(self.population[mom].chromosome[j],self.population[dad].chromosome[j])
            self.population[kid].chromosome[j] += self.mutation_amt * random.gauss(0,1.0)
            if self.population[kid].chromosome[j] > self.ub:
                self.population[kid].chromosome[j] = self.ub
            if self.population[kid].chromosome[j] < self.lb:
                self.population[kid].chromosome[j] = self.lb
            if self.population[kid].chromosome[j] < 0.5:
                self.population[kid].chromosome[j] = 0
            else: self.population[kid].chromosome[j] = 1

        self.population[kid].calculate_fitness()
        for i in range(self.chromosome_length):
            self.hacker_tracker_x.append(self.population[kid].chromosome[i])
        # self.hacker_tracker_y.append(self.population[kid].chromosome[1])
        self.hacker_tracker_z.append(self.population[kid].fitness)
       
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
        print("Best Individual: ",str(best_individual)," ", self.population[best_individual].chromosome, "\nFitness: ", str(best_fitness))
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

#PopSize = 62 # This works
# PopSize = 41
# PopSize = 39
#PopSize = 52
# mu_amt  = 0.00483
# mu_amt  = 0.0048304
# mu_amt = 0.0048312
# mu_amt = 0.00483035
#mu_amt = 0.00482932
# mu_amt = 0.00
# mu_amt = 0.00082
# mu_amt = 0.00084

# PopSize = 53
# mu_amt = 0.000839
# method = 'svml'

PopSize = 50
mu_amt = 0.05
method = 'svml'

maxRun = 10
bestIndividualAllRuns = []
fitnessAllRuns = []

start_time = datetime.datetime.now()
assignmentPart = 'SSGA'
f = open(assignmentPart+'_'+method+'_'+str(PopSize)+'_'+str(mu_amt)+'.txt', "w")
f.write('Assignment Part =  {}'.format(assignmentPart))
f.write('\nPopulation size =  {}'.format(str(PopSize)))
f.write('\nmu_amt =  {}'.format(str(mu_amt)))
f.write('\nmethod =  {}\n'.format(method))

for run in range(maxRun):
    simple_steadystate_ga = aSimpleSteadyStateGA(PopSize,ChromLength,mu_amt,lb,ub,method)

    simple_steadystate_ga.generate_initial_population()
    # simple_steadystate_ga.print_population()

    for i in range(MaxEvaluations-PopSize+1):
        simple_steadystate_ga.evolutionary_cycle()
        if (i % PopSize == 0):
            if (plot == 1):
                simple_steadystate_ga.plot_evolved_candidate_solutions()
            #print("At Iteration: " + str(i))
            # simple_steadystate_ga.print_population()
        # if (simple_steadystate_ga.get_best_fitness() > 0.997544): #= 0.9975438): #
        #     break

    # print("\nFinal Population\n")
    # simple_steadystate_ga.print_population()
    simple_steadystate_ga.print_best_max_fitness()
    # print('\n')
    # print("Function Evaluations: " + str(PopSize+i))
    # simple_steadystate_ga.plot_evolved_candidate_solutions()
np_fitnessAllRuns = np.array(fitnessAllRuns)

f.write('\nBest individual in each run - ')
for ele in bestIndividualAllRuns:
    f.write('\n[')
    f.write(', '.join(str(i) for i in ele))
    f.write(']')
print(fitnessAllRuns, np.max(np_fitnessAllRuns), np.mean(np_fitnessAllRuns))
f.write('\n')
f.write(str(fitnessAllRuns))
f.write('\nBest fitness over all runs =  {}'.format(str(np.max(np_fitnessAllRuns))))
f.write('\nMean fitness over all runs = {}'.format(str(np.mean(np_fitnessAllRuns))))
print('Best individual among all runs is - \n', bestIndividualAllRuns[np_fitnessAllRuns.argmax()])
f.write('\nBest individual among all runs is - \n')
# f.write(','.join(str(bestIndividualAllRuns[np_fitnessAllRuns.argmax()])))
f.write('[')
f.write(', '.join(str(i) for i in bestIndividualAllRuns[np_fitnessAllRuns.argmax()]))
f.write(']')

end_time = datetime.datetime.now()

time_diff = (end_time - start_time)
execution_time = time_diff.total_seconds()/60.0
print('\nExecution time = {} minutes'.format(execution_time))
f.write('\nExecution time = {} minutes'.format(execution_time))
f.close()



    