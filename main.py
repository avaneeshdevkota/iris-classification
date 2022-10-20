import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

#Column Labels
cols = ["SLength", "SWidth", "PLength", "PWidth", "Class"]
df = pd.DataFrame(pd.read_csv("Dataset/iris.data"))
df.columns = cols

#Separate training, validating and testing data
#dp.sample shuffles the data and np.split splits 60% into train, 20% into valid and 20% into test

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def data_scaler(dataframe, undersample = False):

    # x is our features matrix and y is our target
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    #Scaler normalizes the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    #Equalizes the number of samples of different classes by deleting samples of overrepresented classes

    if (undersample == True):

        rus = RandomUnderSampler()
        x,y = rus.fit_resample(x,y)
        
    #Data is the features matrix and target side by side, given by np.hstack

    data = np.hstack((x, np.reshape(y,(-1,1))))

    return data, x, y

#We only wish to check against valid and test so we don't oversample.

train, xtrain, ytrain = data_scaler(train, undersample = True)
valid, xvalid, yvalid = data_scaler(valid, undersample = False)
test, xtest, ytest = data_scaler(test, undersample = False)

#K_Nearest_Neighbours

def k_nearest_neighbours(xtrain, ytrain, xtest, ytest):

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report

    #We compare with the nn closest neighbours.
    nn = 10

    knn_model = KNeighborsClassifier(n_neighbors= nn)
    knn_model.fit(xtrain, ytrain)

    ypredictions = knn_model.predict(xtest)
    report = classification_report(ytest, ypredictions)
    print(report)

k_nearest_neighbours(xtrain, ytrain, xtest, ytest)

