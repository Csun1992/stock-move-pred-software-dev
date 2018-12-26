import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, cluster, model_selection, svm
from sklearn import tree
import sys

# This is a virtual class of classifier
# Other classifiers will inherit from this class and rewrite the train() method

class classifier(object):
    def __init__(self, microDataLoc, clusterNum=1, macroDataLoc="data/clusterData.txt"):
        self.microDataLoc = microDataLoc
        self.macroDataLoc = macroDataLoc
        self.clusterNum = clusterNum

    def cluster(self):
        data = np.loadtxt(self.macroDataLoc)
        cleanData = preprocessing.scale(data)
        kmeans = cluster.KMeans(n_clusters=self.clusterNum, random_state=11).fit(cleanData)
        groupNum = np.array(kmeans.labels_)
        return groupNum

    def prepareData(self):
        data = np.loadtxt(self.microDataLoc)
        groupNum = self.cluster()
        group, label = [], []
        for i in range(self.clusterNum):
            group.append(data[groupNum==i, :-1])
            label.append(data[groupNum==i, -1])
        return (group, label)

    def trainTestSplit(self):
        train, test, trainLabel, testLabel = [], [], [], []
        group, label = self.prepareData()
        for i in range(self.clusterNum):
            trainData, testData, trainLabelData, testLabelData = model_selection.train_test_split(group[i],
                    label[i], test_size=0.3, random_state=11)
            train.append(trainData)
            test.append(testData)
            trainLabel.append(trainLabelData)
            testLabel.append(testLabelData)
        return (train, test, trainLabel, testLabel)
            
    def train(self):
        pass        # virtual method to be overwritten in each specific classifier
                    # must return a tuple of classifier instances, test data and test labels

    def test(self):
        clf, test, testLabel = self.train()
        error = []
        for i in range(self.clusterNum):
            pred = (clf[i].predict(test[i]) == 1)
            caseError = sum([i != j for (i,j) in zip(testLabel[i], pred)])
            error.append(float(caseError)/len(pred))
        return error

    def reportResult(self):
        error = self.test()
        for i in range(self.clusterNum):
            print "group NO." + str(i+1) + " correct rate"
            print 1-error[i]
        print '\n'
        return error

class RandomForest(classifier):
    def __init__(self, microDataLoc, estimators=55, depth=7):
        classifier.__init__(self, microDataLoc)
        self.estimators = estimators
        self.depth = depth

    def train(self):
        train, test, trainLabel, testLabel = self.trainTestSplit()
        clf = [RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, random_state=41) for i in range(self.clusterNum)]
        for i in range(self.clusterNum):
            clf[i].fit(train[i], trainLabel[i])
        return (clf, test, testLabel) # return test and testLabel to self.test() so no need to
                                      # recompute the testing data again



# Object that for svm with clustering 
class svmStockPred(classifier):
    def __init__(self, microDataLoc, clusterNum=1, macroDataLoc="data/clusterData.txt"):
        classifier.__init__(self, microDataLoc, clusterNum=1, macroDataLoc="data/clusterData.txt")

    def train(self):
        train, test, trainLabel, testLabel = self.trainTestSplit()
        clf = [svm.SVC() for i in range(self.clusterNum)]
        for i in range(self.clusterNum):
            clf[i].fit(train[i], trainLabel[i])
        return (clf, test, testLabel) # return test and testLabel to self.test() so no need to
                                      # recompute the testing data again


# StockPredNoClassification class is a class to classify the stock price direction 
# without using clustering. The macro data used for clustering now was combined with
# micro data for classification
class svmNoCluster(svmStockPred):
    def __init__(self, microDataLoc):
        svmStockPred.__init__(self, microDataLoc)

    def prepareData(self):
        group, label = [], []
        microData = np.loadtxt(self.microDataLoc)
        macroData = np.loadtxt(self.macroDataLoc)
        data = np.concatenate((macroData, microData), axis=1)
        for i in range(self.clusterNum):
            group.append(data[: , :-1])
            label.append(data[: , -1])
        return (group, label)

    def reportResult(self):
        error = self.test()
        print "Without Clustering, the correct classification rate is"
        print 1-error[0]
        print '\n'
        return error

            




# This is a virtual class of classifier
# Other classifiers will inherit from this class and rewrite the train() method

class classifier(object):
    def __init__(self, microDataLoc, clusterNum=1, macroDataLoc="data/clusterData.txt"):
        self.microDataLoc = microDataLoc
        self.macroDataLoc = macroDataLoc
        self.clusterNum = clusterNum

    def cluster(self):
        data = np.loadtxt(self.macroDataLoc)
        cleanData = preprocessing.scale(data)
        kmeans = cluster.KMeans(n_clusters=self.clusterNum, random_state=11).fit(cleanData)
        groupNum = np.array(kmeans.labels_)
        return groupNum

    def prepareData(self):
        data = np.loadtxt(self.microDataLoc)
        groupNum = self.cluster()
        group, label = [], []
        for i in range(self.clusterNum):
            group.append(data[groupNum==i, :-1])
            label.append(data[groupNum==i, -1])
        return (group, label)

    def trainTestSplit(self):
        train, test, trainLabel, testLabel = [], [], [], []
        group, label = self.prepareData()
        for i in range(self.clusterNum):
            trainData, testData, trainLabelData, testLabelData = model_selection.train_test_split(group[i],
                    label[i], test_size=0.3, random_state=11)
            train.append(trainData)
            test.append(testData)
            trainLabel.append(trainLabelData)
            testLabel.append(testLabelData)
        return (train, test, trainLabel, testLabel)
            
    def train(self):
        pass        # virtual method to be overwritten in each specific classifier
                    # must return a tuple of classifier instances, test data and test labels

    def test(self):
        clf, test, testLabel = self.train()
        error = []
        for i in range(self.clusterNum):
            pred = (clf[i].predict(test[i]) == 1)
            caseError = sum([i != j for (i,j) in zip(testLabel[i], pred)])
            error.append(float(caseError)/len(pred))
        return error

    def reportResult(self):
        error = self.test()
        for i in range(self.clusterNum):
            print "group NO." + str(i+1) + " correct rate"
            print 1-error[i]
        print '\n'
        return error

class DecisionTreeStockPrediction(classifier):
    def __init__(self, microDataLoc, clusterNum=1, macroDataLoc="data/clusterData.txt"):
        classifier.__init__(self, microDataLoc, clusterNum, macroDataLoc)

    def train(self):
        train, test, trainLabel, testLabel = self.trainTestSplit()
        clf = [tree.DecisionTreeClassifier() for i in range(self.clusterNum)]
        for i in range(self.clusterNum):
            clf[i].fit(train[i], trainLabel[i])
        return (clf, test, testLabel) # return test and testLabel to self.test() so no need to
                                      # recompute the testing data again



class DecisionTreeStockPredictionNoClustering(DecisionTreeStockPrediction):
    def __init__(self, microDataLoc):
        DecisionTreeStockPrediction.__init__(self, microDataLoc)

    def train(self):
        train, test, trainLabel, testLabel = self.trainTestSplit()
        clf = [tree.DecisionTreeClassifier() for i in range(self.clusterNum)]
        for i in range(self.clusterNum):
            clf[i].fit(train[i], trainLabel[i])
        return (clf, test, testLabel) # return test and testLabel to self.test() so no need to
                                      # recompute the testing data again



class RandomForest(classifier):
    def __init__(self, microDataLoc, estimators=55, depth=7):
        classifier.__init__(self, microDataLoc)
        self.estimators = estimators
        self.depth = depth

    def train(self):
        train, test, trainLabel, testLabel = self.trainTestSplit()
        clf = [RandomForestClassifier(n_estimators=self.estimators, max_depth=self.depth, random_state=41) for i in range(self.clusterNum)]
        for i in range(self.clusterNum):
            clf[i].fit(train[i], trainLabel[i])
        return (clf, test, testLabel) # return test and testLabel to self.test() so no need to
                                      # recompute the testing data again



# Object that for svm with clustering 
class svmStockPred(classifier):
    def __init__(self, microDataLoc, clusterNum=1, macroDataLoc="data/clusterData.txt"):
        classifier.__init__(self, microDataLoc, clusterNum=1, macroDataLoc="data/clusterData.txt")

    def train(self):
        train, test, trainLabel, testLabel = self.trainTestSplit()
        clf = [svm.SVC() for i in range(self.clusterNum)]
        for i in range(self.clusterNum):
            clf[i].fit(train[i], trainLabel[i])
        return (clf, test, testLabel) # return test and testLabel to self.test() so no need to
                                      # recompute the testing data again


# StockPredNoClassification class is a class to classify the stock price direction 
# without using clustering. The macro data used for clustering now was combined with
# micro data for classification
class svmNoCluster(svmStockPred):
    def __init__(self, microDataLoc):
        svmStockPred.__init__(self, microDataLoc)

    def prepareData(self):
        group, label = [], []
        microData = np.loadtxt(self.microDataLoc)
        macroData = np.loadtxt(self.macroDataLoc)
        data = np.concatenate((macroData, microData), axis=1)
        for i in range(self.clusterNum):
            group.append(data[: , :-1])
            label.append(data[: , -1])
        return (group, label)

    def reportResult(self):
        error = self.test()
        print "Without Clustering, the correct classification rate is"
        print 1-error[0]
        print '\n'
        return error





def getStockPrice(fileName):
    price = []
    size = []
    with open(fileName) as f:
        if fileName == 'data/apple':
            for line in f:
                item = line.rstrip().split(',')
                price.append(float(item[-2]))
                size.append(float(item[-1]))
        else:
            for line in f:
                item = line.rstrip().split('\t')
                price.append(float(item[-2]))
                size.append(float(item[-1]))
    return price, size[2:-1]

def getInputData(fileName):        
    fileName = 'data/' + fileName
    price, size = getStockPrice(fileName)
    threeMonthMA = [(i+j+k)/3 for i,j,k in zip(price, price[1: ], price[2: ])]
    del threeMonthMA[-1]
    twoMonthMA = [(i+j)/2 for i,j in zip(price[1: ], price[2: ])]
    del twoMonthMA[-1]
    stockReturn = [(j-i)/i for i,j in zip(price[2: ], price[3: ])]
    classification = map(int, [i>0 for i in stockReturn])
    inputData = np.array([price[2:-1], twoMonthMA, threeMonthMA, size, classification]).T
    fileName = fileName + 'TrainData.txt'
    np.savetxt(fileName, inputData)
    return inputData
