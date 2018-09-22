#!/usr/bin/python

#from pylab import plot,show,norm
#from pylab import plot,show,norm
import numpy as np
import sys
from csv import reader, writer
import matplotlib.pyplot as plt

#from sklearn.datasets import make_classification
#from sklearn.datasets import make_blobs
#from sklearn.datasets import make_gaussian_quantiles
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

X_temp = []
Y_temp = []
trainingX_temp = []
trainingY_temp = []
testingX_temp = []
testingY_temp = []

def load_csv(filename):
    samples = list()
    i = 0
    with open(filename, 'rU') as fd:
        csv_reader = reader(fd)
        for row in csv_reader:
            #row.insert(0, '1') # bias
            if i != 0:
                samples.append(row)
                x = []
                x.append(float(row[0]))
                x.append(float(row[1]))
                X_temp.append(x)
                Y_temp.append(int(row[2]))
                if i < 300:
                    trainingX_temp.append(x)
                    trainingY_temp.append(int(row[2]))
                else:
                    testingX_temp.append(x)
                    testingY_temp.append(int(row[2]))
                    
                
            i += 1

    return samples

# pos 0 has feature 1
# pos 1 has feature 2
# pos 2 has true label
def convert_to_float(samples, column):
    for row in samples:
        row[column] = float(row[column].strip())
    return

def SVMLinearKernel(X, Y):
    #https://chrisalbon.com/machine-learning/cross_validation_parameter_tuning_grid_search.html
    parameter_candidates = [
        {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'kernel': ['linear']},
        #{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]
    # Create a classifier object with the classifier and parameter candidates
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
    
    # Train the classifier on data1's feature and target data
    clf.fit(X, Y) 

    print('Kernel:',clf.best_estimator_.kernel, 'Best C:',clf.best_estimator_.C, 'Best score:', clf.best_score_)
    #print('Best C:',clf.best_estimator_.C)
    #print('Best score:', clf.best_score_) 
    """
    # Apply the classifier trained using data1 to data2, and view the accuracy score
    clf.score(X_testing, Y_testing)
    # Train a new classifier using the best parameters found by the grid search
    #svm.SVC(C=0.1, kernel='linear').fit(X_testing, Y_testing).score(X_testing, Y_testing)  
    svm.SVC(kernel='linear', C=0.1).fit(X_training, Y_training).score(X_testing, Y_testing)  
    # View the accuracy score
    print('Best score for data2:', clf.best_score_) 
    # View the best parameters for the model found using grid search
    print('Best C:',clf.best_estimator_.C)
    """
    return clf.best_score_

def SVM_with_Linear_Kernel(X_training, Y_training, X_testing, Y_testing, output):
    trainingScore = SVMLinearKernel(X_training, Y_training)
    testScore = SVMLinearKernel(X_testing, Y_testing)

    temp = []
    temp.append('svm_linear')
    temp.append(trainingScore)
    temp.append(testScore)
    output.writerow(temp)
    return

def SVMPolyKernel(X, Y):
    #https://chrisalbon.com/machine-learning/cross_validation_parameter_tuning_grid_search.html
    parameter_candidates = [
        {'C': [0.1, 1, 3], 'gamma': [0.1, 0.5], 'degree': [4, 5, 6], 'kernel': ['poly']},
        #{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]
    # Create a classifier object with the classifier and parameter candidates
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
    
    # Train the classifier on data1's feature and target data
    clf.fit(X, Y) 

    print('Kernel:',clf.best_estimator_.kernel, 'Best C:',clf.best_estimator_.C, 'Best score:', clf.best_score_)
    #print('Best C:',clf.best_estimator_.C)
    #print('Best score:', clf.best_score_) 

    return clf.best_score_

def SVM_with_Poly_Kernel(X_training, Y_training, X_testing, Y_testing, output):
    trainingScore = SVMPolyKernel(X_training, Y_training)
    testScore = SVMPolyKernel(X_testing, Y_testing)

    temp = []
    temp.append('svm_polynomial')
    temp.append(trainingScore)
    temp.append(testScore)
    output.writerow(temp)
    return

def SVMRBFKernel(X, Y):
    #https://chrisalbon.com/machine-learning/cross_validation_parameter_tuning_grid_search.html
    parameter_candidates = [
        {'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10], 'kernel': ['rbf']},
        #{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]
    # Create a classifier object with the classifier and parameter candidates
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
    
    # Train the classifier on data1's feature and target data
    clf.fit(X, Y) 

    print('Kernel:',clf.best_estimator_.kernel, 'Best C:',clf.best_estimator_.C, 'Best score:', clf.best_score_)
    #print('Best C:',clf.best_estimator_.C)
    #print('Best score:', clf.best_score_) 

    return clf.best_score_


def SVM_with_RBF_Kernel(X_training, Y_training, X_testing, Y_testing, output):
    trainingScore = SVMRBFKernel(X_training, Y_training)
    testScore = SVMRBFKernel(X_testing, Y_testing)

    temp = []
    temp.append('svm_rbf')
    temp.append(trainingScore)
    temp.append(testScore)
    output.writerow(temp)
    return

def myLogisticRegression(X, Y):
    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    cList = [0.1, 0.5, 1, 5, 10, 50, 100]
    max_accuracy = 0
    for i, c in enumerate(cList):
        logis = LogisticRegression(C=c)
        logis.fit(X, Y) 
        accuracy = logis.score(X, Y)
        if accuracy > max_accuracy:
            max_accuracy = accuracy

    print('Kernel: logisticRegression Best score:', max_accuracy)
          
    return max_accuracy

def Logistic_Regression(X_training, Y_training, X_testing, Y_testing, output):
    trainingScore = myLogisticRegression(X_training, Y_training)
    testScore = myLogisticRegression(X_testing, Y_testing)

    temp = []
    temp.append('logistic')
    temp.append(trainingScore)
    temp.append(testScore)
    output.writerow(temp)
    
    return

def kNearestNeighbors(X, Y):
    #http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    #__init__(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)
    neighborsList = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    leafList = [5,10,15,20,25,30,35,40,45,50,55,60]
    max_accuracy = 0
    for i, neighbors in enumerate(neighborsList):
        for j, leaf in enumerate(leafList):
            #print "neighbors=", neighbors, " leaf=", leaf
            knn = KNeighborsClassifier(n_neighbors=neighbors, leaf_size=leaf)
            knn.fit(X, Y) 
            accuracy = knn.score(X, Y)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
            #print('Kernel: k-NearestNeighbors Best score:', accuracy)

    print('Kernel: k-NearestNeighbors Best score:', max_accuracy)
    
    return max_accuracy

def k_Nearest_Neighbors(X_training, Y_training, X_testing, Y_testing, output):
    trainingScore = kNearestNeighbors(X_training, Y_training)
    testScore = kNearestNeighbors(X_testing, Y_testing)

    temp = []
    temp.append('knn')
    temp.append(trainingScore)
    temp.append(testScore)
    output.writerow(temp)
    
    return

def decissionTrees(X, Y):
    #http://scikit-learn.org/stable/modules/tree.html
    maxDepthList = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    minSamplesSplitList = [2,3,4,5,6,7,8,9,10]
    max_accuracy = 0
    for i, maxDepth in enumerate(maxDepthList):
        for j, minSamplesSplit in enumerate(minSamplesSplitList):
            #print "max_depth=", maxDepth, " min_samples_split=", minSamplesSplit
            clf = tree.DecisionTreeClassifier(max_depth=maxDepth, min_samples_split=minSamplesSplit)
            clf.fit(X, Y) 
            accuracy = clf.score(X, Y)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
            #print('Kernel: Decision Trees Best score:', accuracy)
    
    print('Kernel: Decision Trees Best score:', max_accuracy)
    
    return max_accuracy

def Decision_Trees(X_training, Y_training, X_testing, Y_testing, output):
    trainingScore = decissionTrees(X_training, Y_training)
    testScore = decissionTrees(X_testing, Y_testing)

    temp = []
    temp.append('decision_tree')
    temp.append(trainingScore)
    temp.append(testScore)
    output.writerow(temp)
    
    return

def randomForest(X, Y):
    #http://scikit-learn.org/stable/modules/ensemble.html
    maxDepthList = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    minSamplesSplitList = [2,3,4,5,6,7,8,9,10]
    max_accuracy = 0
    for i, maxDepth in enumerate(maxDepthList):
        for j, minSamplesSplit in enumerate(minSamplesSplitList):
            #print "max_depth=", maxDepth, " min_samples_split=", minSamplesSplit
            rfc = RandomForestClassifier(max_depth=maxDepth, min_samples_split=minSamplesSplit)
            rfc.fit(X, Y) 
            accuracy = rfc.score(X, Y)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
            #print('Kernel: Random Forest Best score:', accuracy)
    
    print('Kernel: Random Forest Best score:', max_accuracy)
    
    return max_accuracy

def Random_Forest(X_training, Y_training, X_testing, Y_testing, output):
    trainingScore = randomForest(X_training, Y_training)
    testScore = randomForest(X_testing, Y_testing)

    temp = []
    temp.append('random_forest')
    temp.append(trainingScore)
    temp.append(testScore)
    output.writerow(temp)
    
    return

def main(script, *args):

    if len(sys.argv) != 3:
        print "Error in arguments!"
        sys.exit()
    trainset = load_csv(sys.argv[1])
    columns = len(trainset[0])
    for i in range(columns):
        convert_to_float(trainset, i)
        
    #for x in trainset:
    #    print x


    #http://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
    plt.title("Two informative features, one cluster per class", fontsize='small')
    X = np.array(X_temp, float)
    Y = np.array(Y_temp, int)
    X_training = np.array(trainingX_temp)
    Y_training = np.array(trainingY_temp)
    X_testing = np.array(testingX_temp)
    Y_testing = np.array(testingY_temp)
    #plt.subplot(1, 2, 1)
    #plt.scatter(X_training[:, 0], X_training[:, 1], c=Y_training, cmap=plt.cm.get_cmap('Blues'))
    #plt.show()
    """
    for C in ((0.1, 0.5, 1, 5, 10, 50, 100)):
    #C = 0.1  # SVM regularization parameter
    #C = 100 # SVM regularization parameter
        plt.subplot(1, 2, 1)
        plt.scatter(X_training[:, 0], X_training[:, 1], c=Y_training, cmap=plt.cm.get_cmap('Blues'))
        svc = svm.SVC(kernel='linear', C=C).fit(X_training, Y_training)
    #rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_training, Y_training)
    #poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_training, Y_training)
        #lin_svc = svm.LinearSVC(C=C).fit(X_training, Y_training)
    
        h = .02  # step size in the mesh
    # create a mesh to plot in
        x_min, x_max = X_training[:, 0].min() - 1, X_training[:, 0].max() + 1
        y_min, y_max = X_training[:, 1].min() - 1, X_training[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    
    #for i, clf in enumerate((svc, lin_svc)):
    #for i, clf in enumerate((lin_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        #plt.subplot(2, 2, i + 1)
        #plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.subplot(1, 2, 2)
        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap('RdBu'), alpha=0.8)

    # Plot also the training points
        plt.scatter(X_training[:, 0], X_training[:, 1], c=Y_training, cmap=plt.cm.get_cmap('Blues'))
    #plt.xlabel('Sepal length')
    #plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title("C=" + str(C))
        #plt.show()
    
    
    #clf = svm.SVC()
    #clf.fit(X_training, Y_training)  
    #print clf
    
    
    #clf.predict([[2., 2.]])
    #print clf
    #print clf.support_vectors_
    #print clf.support_ 
    #print clf.n_support_
    """
    
    fd = open(sys.argv[2],'w')
    output = writer(fd)
    
    
    SVM_with_Linear_Kernel(X_training, Y_training, X_testing, Y_testing, output)
    SVM_with_Poly_Kernel(X_training, Y_training, X_testing, Y_testing, output)
    SVM_with_RBF_Kernel(X_training, Y_training, X_testing, Y_testing, output)
    Logistic_Regression(X_training, Y_training, X_testing, Y_testing, output)
    k_Nearest_Neighbors(X_training, Y_training, X_testing, Y_testing, output)
    Decision_Trees(X_training, Y_training, X_testing, Y_testing, output)
    Random_Forest(X_training, Y_training, X_testing, Y_testing, output)

    fd.close()

"""
svm_linear,0.5886287625418061,0.5920398009950248
svm_polynomial,0.7558528428093646,0.6666666666666666
svm_rbf,0.9464882943143813,0.945273631840796
logistic,0.58862876254180607,0.59203980099502485
knn,1.0,1.0
decision_tree,1.0,1.0
random_forest,1.0,1.0

[Executed at: Fri Jun 30 10:12:25 PDT 2017]

output present: passed [2/2]
svm_linear present: passed [2/2]
svm_linear best score: passed [6/6]
svm_linear test score: passed [6/6]
svm_rbf present: passed [2/2]
svm_rbf best score: passed [6/6]
svm_rbf test score: passed [6/6]
knn present: passed [2/2]
knn best score: passed [6/6]
knn test score: passed [6/6]
random_forest present: passed [2/2]
random_forest best score: passed [6/6]
random_forest test score: passed [6/6]
svm_polynomial present: passed [2/2]
svm_polynomial best score: passed [6/6]
svm_polynomial test score: passed [6/6]
logistic present: passed [2/2]
logistic best score: passed [6/6]
logistic test score: passed [6/6]
decision_tree present: passed [2/2]
decision_tree best score: passed [6/6]
decision_tree test score: passed [6/6]
"Classification",100
"""    

    
if __name__ == '__main__':
    main(*sys.argv)

