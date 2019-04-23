## expectation-maximization algorithm
## by Willow Crawford-Crudell

import math
import numpy

## given fileName, open and read file
## extract component model name and performance
def getData(fileName):    
    file = open(fileName, "r")
    fileLines = file.readlines()
    file.close()
    
    data = []
    count = 0
    for line in fileLines[1:]:
        count += 1
        line = line.split(",")
        data_point = line[7:9]
        data.append(data_point)
    print(len(data))
    print(len(fileLines))
    return data       

## select the initial weights, which should be a convex set
## meaning all weights are positive values and sum to 1
def createInitialWeights(data):
    print("createWeights")
    numWeights = len(data[0])
    initial_weights = numpy.random.dirichlet(numpy.ones(numWeights), size = 1)
    initial_weights = initial_weights.tolist()
    
    initial_weights = initial_weights[0]
    return initial_weights
    
                
## calculate the output of an ensemble model
## given a data point, weights, and the component models 
def fullModel(weights, observation):
    
    prob_fullMod = 0
    for mod in range(len(observation)):
        
        prob_given_mod = float(observation[mod])
        prob_fullMod = prob_fullMod + (weights[mod]*prob_given_mod)
     
    return prob_fullMod

## calculate the average log-likelihood that the current weights were
## used given the data
def likelihood(weights, data):
    log_likelihood = 0
    for i in range(len(data)):
        probabilityi_m = fullModel(weights, data[i])
        
        log_prob       = math.log(probabilityi_m)
        log_likelihood = log_likelihood + log_prob
    avg_log = log_likelihood/len(data)
    return avg_log


## this function runs the EM algorithm. This algorithm:
## 1) calculates the expected amount of times each model is used
##    given the data we have
## 2) updates the weights based on these expected values
## 3) iterates until the change in the average likelihood is very small
def EM_algorithm(weights, data):
    epsilon = 0.01
    delta = 1
    numModels = len(data[0])
    count = 0
    while delta > epsilon:
        count += 1
        newWeights = []
        for x in range(numModels):
            newWeights.append(0) # create a list of weights to update
                                 # we can think of newWeights as the weights
                                 # at interation t

        ## calculate the new weight for each model_m
        for mod in range(numModels):
            expectSum = 0
            
            for i in range(len(data)):
                ## calculate the expected amount of times model m was used
                ## given data point z_i
                
                num  = float(data[i][mod])
                denom     = fullModel(weights, data[i])
                
                expectSum = expectSum + (num/denom)

            ## update the weights of each model_m
            newWeight = expectSum/len(data)
            newWeights[mod] = newWeight

        ## calculate the change in the likelihood value
        likelihoodOld = likelihood(weights, data)
        likelihoodNew = likelihood(newWeights, data)

        weights = newWeights
        print(count, weights)
        delta = abs((likelihoodNew - likelihoodOld)/likelihoodNew)

        ## if delta is not less than epsilon, update weights
        ## (which we can think of as the weights at iteration t-1)
        
    return newWeights

def main():
    ## up to user: specify models, data, and initial weights
    fileName   = "expScores.csv"
    data = getData(fileName)
    
    ## choose initial weights randomly
    initial_weights = createInitialWeights(data)

    finalWeights = EM_algorithm(initial_weights, data)
    print(finalWeights)

main()
