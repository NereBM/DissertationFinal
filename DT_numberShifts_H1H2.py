# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:55:20 2019

@author: NereBM
"""

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
from sklearn.model_selection import train_test_split #to easily split data in train and test
from sklearn.model_selection import KFold
import io
import os
from scipy import misc
import pandas as pd
import numpy as np
import pydotplus
import matplotlib.pyplot as plt

#### metrics for performance evaluation ####
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#RMSE:
#source:https://www.kaggle.com/shotashimizu/09-decisiontree-gridsearchcv
from sklearn.metrics import mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    ''' Root mean squared error regression loss
    
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Estimated target values.
    #source:https://www.kaggle.com/shotashimizu/09-decisiontree-gridsearchcv

    '''
    return np.sqrt(mean_squared_error(y_true, y_pred))

#####################-----   MAIN  -----##########################
fileDir = os.path.dirname(os.path.abspath(__file__))#to get the file path
dataset_149087 = pd.read_csv(os.path.join(fileDir, 'DT_datasets/dataset_149087.csv'), sep = ',')

#....................... Last preprocessing of the data ....................
#checking missing values:
dataset_149087.apply(lambda x: sum(x.isna()))
dataset_149087[dataset_149087['8'].isna()]['Weekday'].value_counts()
dataset_149087[dataset_149087['21'].isna()]['Weekday'].value_counts()
#The opening hours for this store are: Mon - Sat: 7h-21h and Sun: 8h-20h so it makes sense that for Sundays we have a lot
#of missing values

#For missing values for 8h and 21h for sundays and 8h for Saturdays I replace them for 0, because the most likely
#reason is that there's no demand:
mask = (dataset_149087['8'].isna()) & ((dataset_149087.Weekday =='Saturday') |(dataset_149087.Weekday =='Sunday'))
dataset_149087.loc[mask, '8'] = 0 #set the values to zero
mask2 = (dataset_149087['21'].isna()) & (dataset_149087.Weekday =='Sunday')
dataset_149087.loc[mask2, '21'] = 0

#For the rest of missing values, maybe it's better to delete the whole row as there are not a lot of them and we don't know
#for sure if there was no demand or if there was demand but the value is missing. For the moment we are goint to set all of them to zeros:
dataset_149087.loc[dataset_149087['8'].isna(), '8'] = 0
dataset_149087.loc[dataset_149087['9'].isna(), '9'] = 0
dataset_149087.loc[dataset_149087['10'].isna(), '10'] = 0
dataset_149087.loc[dataset_149087['11'].isna(), '11'] = 0
dataset_149087.loc[dataset_149087['12'].isna(), '12'] = 0
dataset_149087.loc[dataset_149087['13'].isna(), '13'] = 0
dataset_149087.loc[dataset_149087['14'].isna(), '14'] = 0
dataset_149087.loc[dataset_149087['15'].isna(), '15'] = 0
dataset_149087.loc[dataset_149087['16'].isna(), '16'] = 0
dataset_149087.loc[dataset_149087['17'].isna(), '17'] = 0
dataset_149087.loc[dataset_149087['18'].isna(), '18'] = 0
dataset_149087.loc[dataset_149087['19'].isna(), '19'] = 0
dataset_149087.loc[dataset_149087['20'].isna(), '20'] = 0
dataset_149087.loc[dataset_149087['21'].isna(), '21'] = 0
#check that there are no missing values left
dataset_149087.apply(lambda x: sum(x.isna()))

#transform categorical attributes
dataset_149087.Weekday.value_counts()
#IMPORTANT: NUMBERS NOW ARE THE SAME FORMAT THAT THE INPUT OF THE OPTIMISATION ALGORITHM(Wed=0, Tue=6)
categorical_to_num = {"Weekday":     {"Monday": 5, "Tuesday": 6, "Wednesday": 0, "Thursday": 1, "Friday": 2, "Saturday": 3, "Sunday": 4}}
dataset_149087.replace(categorical_to_num, inplace=True)
dataset_149087['Weekday']=pd.to_numeric(dataset_149087.Weekday)

#selecting the weeks for the test
W33 = ['2018-08-15','2018-08-16','2018-08-17','2018-08-18','2018-08-19','2018-08-20','2018-08-21']
W4 = ['2019-01-23','2019-01-24','2019-01-25','2019-01-26','2019-01-27','2019-01-28','2019-01-29']
#the following week is the easter holidays, but the 21 is not in the data so cannot use it
#W16 = ['2019-04-17','2019-04-18','2019-04-19','2019-04-20','2019-04-21','2019-04-22','2019-04-23']
W15 = ['2019-04-10','2019-04-11','2019-04-12','2019-04-13','2019-04-14','2019-04-15','2019-04-16']
W46 = ['2018-11-14','2018-11-15','2018-11-16','2018-11-17','2018-11-18','2018-11-19','2018-11-20']

#here data is separated into training and test datasets. We leave out of the training test(which we will
#use for parameter tunning)
mask_weeks_out = (~dataset_149087['Date_x'].isin(W33))&(~dataset_149087['Date_x'].isin(W4))&(~dataset_149087['Date_x'].isin(W15))&(~dataset_149087['Date_x'].isin(W46))
dataset_149087_training=dataset_149087[mask_weeks_out]
dataset_149087_test=dataset_149087[~mask_weeks_out]

#deleting the date from the dataset
dataset_149087_training = dataset_149087_training.drop(['Date_x'], axis=1)
features_149087 = ["Weekday","8","9","10","11","12","13","14","15","16","17","18","19","20","21"]
labels_149087 = ["Total_overlap_4","Total_overlap_3","Total_overlap_2","Starting_time_ov_4","Starting_time_ov_3","Starting_time_ov_2"]

########### Using DT with classification ##############

def kfold_cross_validation(df,tree,folds_num,features,target):
    kf = KFold(n_splits = folds_num, shuffle=True)
    attributes = df[features]
    labels = df[target]
    accuracy= []
    precision = []
    recall = []
    f1score =[]
    #RMSE = []
    scores= []
    for i in range(folds_num):
        result = next(kf.split(attributes), None)
        x_train = attributes.iloc[result[0]]
        x_test = attributes.iloc[result[1]]
        y_train = labels.iloc[result[0]]
        y_test = labels.iloc[result[1]]
        model = tree.fit(x_train,y_train)
        y_pred = tree.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average="weighted")) #labels=np.unique(y_pred) can be added to calculate the measure only for the labels that have predicted samples
        recall.append(recall_score(y_test, y_pred, average="weighted"))
        f1score.append(f1_score(y_test, y_pred, average="weighted"))
        #RMSE.append(root_mean_squared_error(y_test, y_pred))
        #accuracy.append(model.score(x_test,y_test))
    #print("Accuracy:",accuracy)
    #print("Avg accuracy:",np.mean(accuracy))
    scores = [np.mean(accuracy),np.mean(precision), np.mean(recall),np.mean(f1score)]
    #scores = [np.mean(accuracy),np.mean(precision), np.mean(recall),np.mean(f1score),np.mean(RMSE)]
    return(scores)

def DT_param_tunning_classification(df,features,target):
    max_score = 0
    score = 0
    criterion = ["gini","entropy"]
    splitter = ["best", "random"]
    max_depth = [None,2,3,4,5,6,7,8,9,10]
    min_samples_split = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30]
    min_samples_leaf = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,20]
    max_features = [None,"auto","sqrt","log2"]
    for crit in criterion:
        for split in splitter:
            for depth in max_depth:
                for min_split in min_samples_split:
                    for min_leaf in min_samples_leaf:
                        for feat in max_features:
                            tree = DecisionTreeClassifier(criterion =crit ,splitter=split,max_depth=depth,min_samples_split=min_split,min_samples_leaf=min_leaf,max_features=feat)
                            print("Criterion: "+str(crit)+", Splitter: "+str(split)+", max_depth: "+str(depth)+", min_samples_split: "+str(min_split)+", min_samples_leaf: "+str(min_leaf)+", max_features: "+str(feat))
                            score = kfold_cross_validation(df,tree,10,features,target)
                            #result="Criterion:,"+str(crit)+", Splitter:,"+str(split)+", max_depth:,"+str(depth)+", min_samples_split:,"+str(min_split)+", min_samples_leaf:,"+str(min_leaf)+", max_features:,"+str(feat)+",score:,"+str(score)+",\n"
                            result="Criterion:,"+str(crit)+", Splitter:,"+str(split)+", max_depth:,"+str(depth)+", min_samples_split:,"+str(min_split)+", min_samples_leaf:,"+str(min_leaf)+", max_features:,"+str(feat)+",accuracy:,"+str(score[0])+",precision:,"+str(score[1])+",recall:,"+str(score[2])+",f1score:,"+str(score[3])+",\n"
                            file1 = open("Results/Num_shifts/DT_param_tunning_NumShifts_class.txt","a") #Append Only (‘a’) : Open the file for writing. The file is created if it does not exist. The handle is positioned at the end of the file. The data being written will be inserted at the end, after the existing data.
                            file1.write(result)
                            file1.close()
                            #if(score>max_score):
                            if(score[0]>max_score):#here we use accuracy, but we can change it for example to f1score using score[3]
                                max_score=score[0]
                                max_score_param = "Criterion: "+str(crit)+", Splitter: "+str(split)+", max_depth: "+str(depth)+", min_samples_split: "+str(min_split)+", min_samples_leaf: "+str(min_leaf)+", max_features: "+str(feat)
    best = [max_score,max_score_param]
    return(best)
#parameter tunning: we use as input data the dataset without the weeks we're going to predict later
best_DT_class = DT_param_tunning_classification(dataset_149087_training,features_149087,"Num_shifts")
#Result: [0.8416666666666668,
# 'Criterion: entropy, Splitter: random, max_depth: 8, min_samples_split: 16, min_samples_leaf: 16, max_features: None']


############ Using DT with regression ##################

def kfold_cv_regression(df,tree,folds_num,features,target):
    kf = KFold(n_splits = folds_num, shuffle=True)
    attributes = df[features]
    labels = df[target]
    RMSE = []
    for i in range(folds_num):
        result = next(kf.split(attributes), None)
        x_train = attributes.iloc[result[0]]
        x_test = attributes.iloc[result[1]]
        y_train = labels.iloc[result[0]]
        y_test = labels.iloc[result[1]]
        model = tree.fit(x_train,y_train)
        y_pred = tree.predict(x_test)
        RMSE.append(root_mean_squared_error(y_test, y_pred))
    #print("RMSE:",RMSE)
    #print("Avg RMSE:",np.mean(RMSE))
    return(np.mean(RMSE))
    
def DT_param_tunning_regression(df,features,target):
    min_RMSE = 0
    RMSE = 0
    criterion = ["mse","friedman_mse","mae"]
    splitter = ["best", "random"]
    max_depth = [None,2,3,4,5,6,7,8,9,10]
    min_samples_split = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30]
    min_samples_leaf = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,20]
    max_features = [None,"auto","sqrt","log2"]
    for crit in criterion:
        for split in splitter:
            for depth in max_depth:
                for min_split in min_samples_split:
                    for min_leaf in min_samples_leaf:
                        for feat in max_features:
                            tree = DecisionTreeRegressor(criterion =crit ,splitter=split,max_depth=depth,min_samples_split=min_split,min_samples_leaf=min_leaf,max_features=feat)
                            print("Criterion: "+str(crit)+", Splitter: "+str(split)+", max_depth: "+str(depth)+", min_samples_split: "+str(min_split)+", min_samples_leaf: "+str(min_leaf)+", max_features: "+str(feat))
                            RMSE = kfold_cv_regression(df,tree,10,features,target)
                            print(score)
                            #result="Criterion:,"+str(crit)+", Splitter:,"+str(split)+", max_depth:,"+str(depth)+", min_samples_split:,"+str(min_split)+", min_samples_leaf:,"+str(min_leaf)+", max_features:,"+str(feat)+",score:,"+str(score)+",\n"
                            result="Criterion:,"+str(crit)+", Splitter:,"+str(split)+", max_depth:,"+str(depth)+", min_samples_split:,"+str(min_split)+", min_samples_leaf:,"+str(min_leaf)+", max_features:,"+str(feat)+",RMSE:,"+str(score)+",\n"
                            file1 = open("Results/Num_shifts/DT_param_tunning_NumShifts_reg_2.txt","a") #Append Only (‘a’) : Open the file for writing. The file is created if it does not exist. The handle is positioned at the end of the file. The data being written will be inserted at the end, after the existing data.
                            file1.write(result)
                            file1.close()
                            #if(score>max_score):
                            if(RMSE<min_RMSE):
                                min_RMSE=RMSE
                                min_RMSE_param = "Criterion: "+str(crit)+", Splitter: "+str(split)+", max_depth: "+str(depth)+", min_samples_split: "+str(min_split)+", min_samples_leaf: "+str(min_leaf)+", max_features: "+str(feat)
    best = [min_RMSE,min_RMSE_param]
    return(best)
    
#PENDIENTE DE EJECUTAR:
best_DT_reg = DT_param_tunning_regression(dataset_149087_training,features_149087,"Num_shifts")
#[0.7962210814217162,
# 'Criterion: friedman_mse, Splitter: best, max_depth: 9, min_samples_split: 3, min_samples_leaf: 1, max_features: None']





##################  Using ANN ##############################
from sklearn.neural_network import MLPClassifier, MLPRegressor

#one test:
ANN_class = MLPClassifier(alpha=1, max_iter=1000)
score = kfold_cross_validation(dataset_149087_training,ANN_class,10,features_149087,"Num_shifts")

#source: https://www.heatonresearch.com/2017/06/01/hidden-layers.html
#The number of hidden neurons should be between the size of the input layer and the size of the output layer.
#The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
#The number of hidden neurons should be less than twice the size of the input layer.
#In this case our input layer is 15:
#len(features_149087)
"""
def ANN_param_tunning_classification(df,features,target):
    max_score = 0
    score = 0
    #hidden_layer_sizes is a tuple of size (n_layers -2) because input and output layers don't count
    #so: (7,) is one hidden layer of 7 units, and (7,7) is 2 hidden layers of 7 units and (7,7,7) is 3 hidden
    #layers with 7 units
    hidden_layer_sizes = [(5,),(10,),(12,),(14,),(16,),(20,),(50,),(100,),
                          (5,2),(10,5),(12,6),(14,7),(16,8),(20,10),(50,25),(100,50),
                          (5,5),(10,10),(12,12),(14,14),(16,16),(20,20),(50,50),(100,100)]
    activation  = ["relu","identity", "logistic", "tanh"]
    solver = ["lbfgs", "sgd", "adam"]
    alpha = [0.00001,0.0001,0.0002,0.0003,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]
    max_iter = [100,150,200,250,300,350,400,500,600,700,800,900,1000,1500,2000,5000]
    early_stopping  = [True,False]
    learning_rate = ["constant", "invscaling", "adaptive"]
    for hid in hidden_layer_sizes:
        for act in activation:
            for sol in solver:
                for al in alpha:
                    for max_it in max_iter:
                        for es in early_stopping:
                            for lr in learning_rate:
                                ANN = MLPClassifier(hidden_layer_sizes= hid,activation =act ,solver=sol,alpha=al,max_iter=max_it,early_stopping=es,learning_rate=lr)
                                print("hidden_layer_sizes"+str(hid)+"Activation: "+str(act)+", Solver: "+str(sol)+", alpha: "+str(al)+", max_iter: "+str(max_it)+", early_stopping: "+str(es)+", learning_rate: "+str(lr))
                                score = kfold_cross_validation(df,ANN,10,features,target)
                                #result="Criterion:,"+str(crit)+", Splitter:,"+str(split)+", max_depth:,"+str(depth)+", min_samples_split:,"+str(min_split)+", min_samples_leaf:,"+str(min_leaf)+", max_features:,"+str(feat)+",score:,"+str(score)+",\n"
                                result="hidden_layer_sizes"+str(hid)+"Activation:,"+str(act)+", Solver:,"+str(sol)+", alpha:,"+str(al)+", max_iter:,"+str(max_it)+", early_stopping:,"+str(es)+", learning_rate:,"+str(lr)+",accuracy:,"+str(score[0])+",precision:,"+str(score[1])+",recall:,"+str(score[2])+",f1score:,"+str(score[3])+",\n"
                                file1 = open("Results/Num_shifts/ANN_param_tunning_NumShifts_class.txt","a") #Append Only (‘a’) : Open the file for writing. The file is created if it does not exist. The handle is positioned at the end of the file. The data being written will be inserted at the end, after the existing data.
                                file1.write(result)
                                file1.close()
                                print("hidden_layer_sizes"+str(hid)+"Activation:,"+str(act)+", Solver:,"+str(sol)+", alpha:,"+str(al)+", max_iter:,"+str(max_it)+", early_stopping:,"+str(es)+", learning_rate:,"+str(lr))
                                #if(score>max_score):
                                if(score[0]>max_score):#here we use accuracy, but we can change it for example to f1score using score[3]
                                    max_score=score[0]
                                    max_score_param = "hidden_layer_sizes"+str(hid)+"Activation:,"+str(act)+", Solver:,"+str(sol)+", alpha:,"+str(al)+", max_iter:,"+str(max_it)+", early_stopping:,"+str(es)+", learning_rate:,"+str(lr)
    best = [max_score,max_score_param]
    return(best)
    
best_ANN_class = ANN_param_tunning_classification(dataset_149087_training,features_149087,"Num_shifts")
"""

############### Retrieving the data from the left-out weeks #############################

W33 = ['2018-08-15','2018-08-16','2018-08-17','2018-08-18','2018-08-19','2018-08-20','2018-08-21']
W4 = ['2019-01-23','2019-01-24','2019-01-25','2019-01-26','2019-01-27','2019-01-28','2019-01-29']
#the following week is the easter holidays, but the 21 is not in the data so cannot use it
#W16 = ['2019-04-17','2019-04-18','2019-04-19','2019-04-20','2019-04-21','2019-04-22','2019-04-23']
W15 = ['2019-04-10','2019-04-11','2019-04-12','2019-04-13','2019-04-14','2019-04-15','2019-04-16']
W46 = ['2018-11-14','2018-11-15','2018-11-16','2018-11-17','2018-11-18','2018-11-19','2018-11-20']

W1 = dataset_149087[dataset_149087['Date_x'].isin(W33)]
W2 = dataset_149087[dataset_149087['Date_x'].isin(W46)]
W3 = dataset_149087[dataset_149087['Date_x'].isin(W4)]
W4 =dataset_149087[dataset_149087['Date_x'].isin(W15)]

##### Heuristics to calculate the demand (old way) #######
#---------------  Heuristic 1 -----------------------------
hours = ['8','9','10','11','12','13','14','15','16','17','18','19','20','21']

W1[hours].apply(lambda x: x/10, axis=0)
W1[hours].apply(lambda x: round(x/10,0), axis=0)
W2[hours].apply(lambda x: x/10, axis=0)
W2[hours].apply(lambda x: round(x/10,0), axis=0)
W3[hours].apply(lambda x: x/10, axis=0)
W3[hours].apply(lambda x: round(x/10,0), axis=0)
W4[hours].apply(lambda x: x/10, axis=0)
W4[hours].apply(lambda x: round(x/10,0), axis=0)

#-----------------  Heuristic 2 ----------------------------
def transform_demand_by_experience (n):
    p=0
    print(n)
    if n<20:
        p=1
    elif (n>=20) & (n<45):
        p=2
    elif (n>=45) & (n<60):
        p=3
    elif (n>=60) & (n<100):
        p=4
    elif n>=100:
        p=5
    return p

#applymap(applies a function elementwise on a dataframe)
W1[hours].applymap(transform_demand_by_experience)
W2[hours].applymap(transform_demand_by_experience)
W3[hours].applymap(transform_demand_by_experience)
W4[hours].applymap(transform_demand_by_experience)


#######################      Number of shifts    ###########################################
#################### PREDICTIONS USING THE PARAMETERS OBTAINED ##################
tree_class = DecisionTreeClassifier(criterion = 'entropy' ,splitter= 'random', max_depth=8, min_samples_split=16, min_samples_leaf=16,max_features=None)
X_train = dataset_149087_training[features_149087]
Y_train = dataset_149087_training["Num_shifts"]

X_test = dataset_149087_test[features_149087]
Y_test = dataset_149087_test["Num_shifts"]

#-----------------          DT classifier          ---------------------------------
model = tree_class.fit(X_train, Y_train) #training

def show_tree(tree, features, path):
    #source: https://www.youtube.com/watch?v=XDbj6PxaSf0
    f = io.StringIO() #This creates an object StringBuffer which is quite similar to
    #a string but, while a string isa fixed-length, immutable character sequences and this is
    #growable and writable character sequences. In this case the object is empty.
    export_graphviz(tree, out_file = f, feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img=misc.imread(path)
    plt.rcParams["figure.figsize"]=(20,20)
    plt.imshow(img)
    
show_tree(model, features_149087,'dec_tree.png')


y_pred_DTclass = tree_class.predict(X_test)
accuracy_score(Y_test, y_pred_DTclass)
#accuracy: 0.7142857142857143
precision_score(Y_test, y_pred_DTclass, average="weighted") #labels=np.unique(y_pred) can be added to calculate the measure only for the labels that have predicted samples
#precision: 0.5904761904761904 
recall_score(Y_test, y_pred_DTclass, average="weighted")
#recall_score: 0.7142857142857143
f1_score(Y_test, y_pred_DTclass, average="weighted")
#f1_score: 0.6439909297052154
X_test['Date']=dataset_149087_test['Date_x']
X_test['DT_class_pred'] = y_pred_DTclass.copy()
X_test.to_csv("Results/Num_shifts/Predictions/DT_class_pred.csv", index = False)

#-----------------          DT regressor          ---------------------------------
tree_reg = DecisionTreeRegressor(criterion ='mse' ,splitter='random', max_depth=3, min_samples_split=20,min_samples_leaf=16,max_features='auto')
model_reg = tree_reg.fit(X_train, Y_train) #training
show_tree(model_reg, features_149087,'dec_tree_reg.png')
X_test_reg = dataset_149087_test[features_149087]
y_pred_DTreg = tree_reg.predict(X_test_reg)
root_mean_squared_error(Y_test, y_pred_DTreg)
X_test['DT_reg_pred'] = np.round(y_pred_DTreg, 0)
X_test.to_csv("Results/Num_shifts/Predictions/DT_reg_pred.csv", index = False)
# RMSE: 0.6396700215221117

