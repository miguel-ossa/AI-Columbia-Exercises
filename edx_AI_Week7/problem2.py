#!/usr/bin/python

#from pylab import plot,show,norm
#from pylab import plot,show,norm
#import numpy
import sys
from csv import reader, writer
#from sklearn import preprocessing
from decimal import *

betas = []
betas.append(0.0)
betas.append(0.0)
betas.append(0.0)


def load_csv(filename):
    samples = list()
    with open(filename, 'r') as fd:
        csv_reader = reader(fd)
        for row in csv_reader:
            #row.insert(0, '1') # bias
            samples.append(row)

    return samples

# pos 0 has feature 1
# pos 1 has feature 2
# pos 2 has true label
def convert_to_float(samples, column):
    for row in samples:
        row[column] = float(row[column].strip())
    return

def mean(rows):
    #total = 0
    #for x in rows:
    #    total += x
    #elements = len(rows)
    #return total / elements
    return sum(rows) / float(len(rows))

def funct(rows):
    m = mean(rows)
    diff = sum((i-m)**2 for i in rows)
    #diff = sum(i**2 for i in rows)
    return diff

def stdev(rows):
    """Calculates the population standard deviation"""
    dev = funct(rows) / float(len(rows))
    return dev**0.5

def scale(samples):
    trainset = []
    #xraw = []
    for row in samples:
        features = []
        features.append(row[0])
        features.append(row[1])
        #z1 = (row[0] - mean(features)) / stdev(features)
        #z2 = (row[1] - mean(features)) / stdev(features)
        z1 = row[0] / stdev(features)
        z2 = row[1] / stdev(features)
        #print "custom = ", z1, z2
        par = []
        par.append(1.0) # bias
        par.append(z1) # feature 1 scaled
        par.append(z2) # feature 2 scaled
        par.append(row[2]) # label
        trainset.append(par)
        #
        #xpar = []
        #xpar.append(row[0])
        #xpar.append(row[1])
        #xraw.append(xpar)
    """
    norm = preprocessing.normalize(xraw)
    trainset = []
    i = 0;
    for row in samples:
        par = []
        par.append(1.0) # bias
        par.append(norm[i][0]) # feature 1 scaled
        par.append(norm[i][1]) # feature 2 scaled
        par.append(row[2]) # label
        trainset.append(par)
        
    print "sklearn = ", preprocessing.normalize(xraw)
    #trainset = preprocessing.scale(xraw)
    """
    return trainset

def f(features):
    """Receives the betas (betas) and one row of features, plus label (features).
    """
    result = 0.0
    i = 0
    for x in features[:-1]: # ignore the label
        result += betas[i] * x
        i += 1
    return result

def gradient_descent(alpha, features):
    """Receives the betas (betas) and one row of features, plus label (features).
    """
    tlabel = features[len(features)-1]
    n = float(len(features) - 1)
    f_x = f(features)
    i = 0
    for x in features[:-1]:
        betas[i] = betas[i] - alpha * 1.0 / n * (f_x - tlabel) * x
        i += 1 
    return
    #return summa

def risk(features):
    """Receives the betas (betas)  and one row of features, plus label (features)
    """
    f_x = f(features)
    tlabel = features[len(features)-1]
    summa = Decimal(0)
    for x in features[:-1]:
        partial = Decimal(f_x - tlabel)
        #print partial
        summa = partial**Decimal(2.0) # minus label
    n = Decimal(len(features) - 1)
    return Decimal(summa * 1 / (2*n))

def main(script, *args):

    if len(sys.argv) != 3:
        print "Error in arguments!"
        sys.exit()
    trainset = load_csv(sys.argv[1])
    columns = len(trainset[0])
    for i in range(columns):
        convert_to_float(trainset, i)

    scaled_trainset = scale(trainset) # bias, features plus label

    fd = open(sys.argv[2],'w')
    output = writer(fd)

    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 0.55]
    #min_risk = Decimal(9999.0)
    for alpha in learning_rates:
        iterations = 0
        risk_betas = Decimal(0)
        ant_risk = Decimal(9999.0)
        #betas = [0.0 for _ in range(len(scaled_trainset[0])-1)]
        betas[0] = 0.0
        betas[1] = 0.0
        betas[2] = 0.0
        convergence = False
        while iterations < 100 and not convergence:
            iterations += 1
            #if iterations == 99 and alpha != 0.55:
            #    convergence = True
            #if risk_betas == ant_risk and alpha == 0.55:
            #    convergence = True
            #    print "convergence with alpha=", alpha, "iterations=", iterations - 1, "risk=", risk_betas
            #else:
            #    ant_risk = risk_betas
            for row in scaled_trainset:
                risk_betas = risk(row)
                gradient_descent(alpha, row)
        #if risk_betas < min_risk:
        #    print "***alpha=", alpha, " risk=", risk_betas
        #    min_risk = risk_betas
        temp = []
        temp.append(alpha)
        temp.append(iterations)
        temp.append(betas[0])
        temp.append(betas[1])
        temp.append(betas[2])
        output.writerow(temp)
    fd.close()

"""

[Executed at: Thu Jun 22 10:47:47 PDT 2017]

alpha = 0.001: alpha passed [1/1]
alpha = 0.001: iterations passed [1/1]
alpha = 0.001: b_intercept failed [0/1]
alpha = 0.001: b_age failed [0/1]
alpha = 0.001: b_weight failed [0/1]
alpha = 0.005: alpha passed [1/1]
alpha = 0.005: iterations passed [1/1]
alpha = 0.005: b_intercept failed [0/1]
alpha = 0.005: b_age failed [0/1]
alpha = 0.005: b_weight failed [0/1]
alpha = 0.01: alpha passed [1/1]
alpha = 0.01: iterations passed [1/1]
alpha = 0.01: b_intercept failed [0/1]
alpha = 0.01: b_age failed [0/1]
alpha = 0.01: b_weight failed [0/1]
alpha = 0.05: alpha passed [1/1]
alpha = 0.05: iterations passed [1/1]
alpha = 0.05: b_intercept failed [0/1]
alpha = 0.05: b_age failed [0/1]
alpha = 0.05: b_weight failed [0/1]
alpha = 0.1: alpha passed [1/1]
alpha = 0.1: iterations passed [1/1]
alpha = 0.1: b_intercept failed [0/1]
alpha = 0.1: b_age failed [0/1]
alpha = 0.1: b_weight failed [0/1]
alpha = 0.5: alpha passed [1/1]
alpha = 0.5: iterations passed [1/1]
alpha = 0.5: b_intercept failed [0/1]
alpha = 0.5: b_age failed [0/1]
alpha = 0.5: b_weight failed [0/1]
alpha = 1: alpha passed [1/1]
alpha = 1: iterations passed [1/1]
alpha = 1: b_intercept failed [0/1]
alpha = 1: b_age failed [0/1]
alpha = 1: b_weight failed [0/1]
alpha = 5: alpha passed [1/1]
alpha = 5: iterations passed [1/1]
alpha = 5: b_intercept passed [1/1]
alpha = 5: b_age passed [1/1]
alpha = 5: b_weight passed [1/1]
alpha = 10: alpha passed [1/1]
alpha = 10: iterations passed [1/1]
alpha = 10: b_intercept passed [1/1]
alpha = 10: b_age passed [1/1]
alpha = 10: b_weight passed [1/1]
alpha = free: alpha passed [1/1]
alpha = free: iterations passed [1/1]
alpha = free: b_intercept failed [0/1]
alpha = free: b_age failed [0/1]
alpha = free: b_weight failed [0/1]
"Linear Regression",26 
"""

if __name__ == '__main__':
    main(*sys.argv)

