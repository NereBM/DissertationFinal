The python packages to be installed to run the code are:
sys
os
pandas
numpy
pandas
datetime
sklearn
matplotlib
seaborn
time


Info by file name:
............................... Final_code.py ............................................
The main file is Final_code.py. In this file, from line 30 to line 68 the inputs must be set. The details of the inputs (their meaning, the formats,etc.) are in these lines in the code.
From line 68 to line 331 the different auxiliar functions are define, each of them with a description of what they do and what they return.
From lines 363 to 440: Pre-processing
From 440 to 460: Dataset creation
From 460 to 668: Creation of the Machine earning models. The parameters used are the parameters obtained as a result of the parameter tuning. The predictions are obtained in this part of the code.
From 668 to 751: Transformation of the predictions to a format suitable for the optimisation algorithm. The files with the results will be saved in Results\Predictions. The following two files contain the inputs for the GA:

Number_shifts_’STOREID’-> contains the input MAX_NUMBER_OF_WORKERS_IN_A_DAY, where “STOREID” will be the corresponding StoreID
Schedule(DEMAND)_’STOREID’WN-> contains the input DEMAND, where “STOREID” will be the corresponding StoreID and WN will be the number N of week(one of the four weeks used as test set)



........................... Preliminary_analysis ...................
It contains a preliminary analysis of the data:
From line 22 to 82: Auxiliar functions
From 83 to 125 : Reading from files and data formatting
From 124 to 162 : Data pre-processing
From 170 to 224: Important measures, ranges and outliers
From 224 to 295 : Value frequency study (includes shifts histograms)
From 295 to 358: Correlation study and scatterplots
From 360 to 394: Minimum and maximum number of workers by store

.................................DT_numberShifts_H1H2 .....................
It contains the parameter tuning and the cross validation functions for the Decision Tree Regressor and the
Decision Tree Classifier for the label Num_shifts. It contains the resulting trees and the function to
represent them graphically.
It also contais the demand calculated using H1 and H2 for each of the four weeks of the test set. (lines 303-336)

................................DT_server_paramTuning.................
It contains the code for the Decision Tree Regressor and the Decision Tree Classifier for the rest of the labels.
This code was run on the server.


................................RF_server_paramTuning.................
It contains the code for the Random Forest Regressor and the Decision Tree Classifier for the rest of the labels.
This code was run on the server.
