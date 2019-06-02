#!/usr/bin/python

#from pylab import plot,show,norm
from pylab import plot,show,norm
import numpy
import sys
from csv import reader, writer
 
class Perceptron:
    def __init__(self):
        self.w = numpy.random.random(3)
        self.w[0] = 0 # weight for bias
        self.w[1] = 0 # weight for feature 1
        self.w[2] = 0 # weight for feature 2
        self.learningRate = 0.003

    def calculate(self, data, output):
        done = False
        while not done:
            sumErrors = 0.0
            for x in data:
                r = x[0]*self.w[0]+x[1]*self.w[1]+x[2]*self.w[2] 
                if x[3] * r <= 0.0: # error. fix the weights
                    self.w[0] += x[3] * x[0]
                    self.w[1] += x[3] * x[1]
                    self.w[2] += x[3] * x[2]
                    sumErrors += 1
            temp = []
            temp.append(int(self.w[1])) # weight for feature 1
            temp.append(int(self.w[2])) # weight for feature 2
            temp.append(int(self.w[0])) # weight for bias
            output.writerow(temp)
            #print temp
            if sumErrors == 0.0:
                done = True
        return

def load_csv(filename):
    samples = list()
    with open(filename, 'r') as fd:
        csv_reader = reader(fd)
        for row in csv_reader:
            row.insert(0, '1') # bias
            samples.append(row)
            
    return samples

# pos 0 has feature 1
# pos 1 has feature 2
# pos 2 has true label
def convert_to_float(samples, column):
    for row in samples:
        row[column] = float(row[column].strip())
    return

def plot_trainset(trainset):
    for x in trainset:
        if x[3] == 1.0:
            plot(x[1],x[2],'ob')  
        else:
            plot(x[1],x[2],'or')
    return

def plot_convergence(trainset, w):
    #n = norm(w)
    #n1 = norm(trainset)
    #ww = w/n
    #wt = trainset/n1
    #ww1 = [ww[2]*wt[2],-ww[1]*wt[1]]
    #ww2 = [-ww[2]*wt[2],ww[1]*wt[1]]
    #plot([ww1[0], ww2[0]],[ww1[1], ww2[1]],'--k')

    plot([0,-w[0]/w[1]], [-w[0]/w[2],0],'--k')
    return

def main(script, *args):
    
    if len(sys.argv) != 3:
        print "Error in arguments!"
        sys.exit()
    trainset = load_csv(sys.argv[1])
    columns = len(trainset[0])
    for i in range(columns):
        convert_to_float(trainset, i)

    perceptron = Perceptron() 
    fd = open(sys.argv[2],'w')
    output = writer(fd)
    perceptron.calculate(trainset, output)
    fd.close()

    plot_convergence(trainset, perceptron.w)
    plot_trainset(trainset)
    show()


if __name__ == '__main__':
    main(*sys.argv)
    
    