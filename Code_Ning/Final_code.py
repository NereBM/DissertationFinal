# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:20:54 2019

@author: NereBM
"""

import sys
import os
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from time import mktime
import time

#### metrics for performance evaluation ####
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error


fileDir = os.path.dirname(os.path.abspath(__file__))#to get the file path

""" 
..............................INPUTS................................

"""

#INPUT: ID of the store
STOREID = 149087 
#INPUT the four weeks that will be used as test set
W33 = ['2018-08-15','2018-08-16','2018-08-17','2018-08-18','2018-08-19','2018-08-20','2018-08-21']
W4 = ['2019-01-23','2019-01-24','2019-01-25','2019-01-26','2019-01-27','2019-01-28','2019-01-29']
W15 = ['2019-04-10','2019-04-11','2019-04-12','2019-04-13','2019-04-14','2019-04-15','2019-04-16']
W46 = ['2018-11-14','2018-11-15','2018-11-16','2018-11-17','2018-11-18','2018-11-19','2018-11-20']

#INPUT: DEMAND_USING_FORMULA = True will use a given formula to calculate the demand from files
DEMAND_USING_FORMULA=True
#INPUT: The name of the file that contains the forecasted demand (ONLY IF DEMAND_USING_FORMULA = False)
FORECASTED_DEMAND_FILE = 'Forecasted_demand.csv' #this file must be saved in the Data folder
"""
The file must have the following columns that contains the following data type:
OrganisationID              int64
StoreInstanceID             int64
StoreID                     int64
Date                        str('%Y-%m-%d')
Hour                        int64
sales                     float64
units                       int64
"""


#INPUT: START_TIME will define the starting time for the slots of the Genetic Algorithm so that
#The resulting values from this process can be used directly. ONLY INTEGERS!!
STARTING_TIME=8

#INPUT: Name of the files that contain the information related to sales and shifts:
#ALL OF THEM MUST BE SAVED IN THE "Data" FOLDER!
SALES_FILE = 'SalesData_Orgs_329970_339259_763128_766174.csv'#(Only used if DEMAND_USING_FORMULA = True)
SALES_FILE2= 'SalesData_Orgs_780237_780251_781676_791175.csv'#(Only used if DEMAND_USING_FORMULA = False)
SHIFTS_FILE= 'ScheduleShifts_2018-04_2019-05.csv'
SHIFTS_FILE_TIMES = 'ScheduleShifts_2018-04_2019-05_times.csv'

#######################################################################################################3
#...................Auxiliar functions: ...............
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

def get_next_year_demand(weeknum,weekday,year):
    ''' receives as an input the week number, the weekday and the year and calculates the 
    date that corresponds to that week number and that weekday of the next year.
        - weeknum: number of week in the year
        - weekday:
        - year:
    '''
    #WATCH OUT THE DAY OF THE WEEK!! for Timestamp Monday=0, Sunday=6
    if(year==2018):
        weeknum -=1#if we calculate the 2019 date we need to substract 1 to the week number
                   #because it counts one week ahead from calendar for this year. WATCH OUT because
                   #for years such as 2018, when the 1st of July is Monday, this is not necessary
    if(weekday==7):
        weekday=0
    day = time.asctime(time.strptime(str(year+1)+' '+str(weeknum)+' '+str(weekday), '%Y %W %w'))#Monday=1, Sunday=0
    return(datetime.strptime(day, "%a %b %d %H:%M:%S %Y"))

def get_number_employees_store(date,hour,df,store):
    ''' calculates the number of employees that were working for a specific date, hour and store:
        date: date we want the number of employees working on
        hour: hour we want the number of employees working at
        store: store we want the number of employees working in
        df: dataframe where the shifts information is contained
    '''
    return(len(df[(df.TransactionDate == date) & (hour>=df.Hour_start) & (hour<df.Hour_finish) & (df.StoreID==store)]))

def get_month(date):
    '''retunts the month (number) of a specific date'''
    date_format=datetime.strptime(date, "%Y-%m-%d")
    return (date_format.month)

def get_weekday(date):
    '''retunts the weekday (written) of a specific date'''
    date_format=datetime.strptime(date, "%Y-%m-%d")
    wd=date_format.weekday()
    #<expression1> if <condition> else <expression2>
    return("Monday" if wd==0 else ("Tuesday" if wd==1 else
             ("Wednesday" if wd==2 else 
                  ("Thursday" if wd==3 else 
                       ("Friday" if wd==4 else
                            ("Saturday" if wd==5 else "Sunday"))))))

def features_demand_id(data): #input only one store
    """ transform the sales information contained in the input data in a format that allows 
    for using it as features. It returns a dataframe with this format, which has the 
    following columns:
        'ID','Date','Weekday','8','9','10','11','12','13','14','15','16','17','18','19','20','21'
        
        data:dataframe with the sales information
    """
    dates=data.NewDate.unique()
    dates.sort()
    df=DataFrame()
    for date in dates:
        #print(date)
        date_str= str(date)
        ID=date_str[:4]+date_str[5:7]+date_str[8:10]
        subset = data[data.NewDate == date]
        lst=[]
        for i in range(8,22):
            try:
                lst+= [subset[subset.Hour==i]['units'].values[0]]
            except:
                lst+=[""]
        Demand_hours = {'ID' : [ID],
                        'Date': [date],
                        'Weekday' : [subset['Weekday'].iloc[1]],
                        '8': [lst[0]],
                        '9': [lst[1]],
                        '10': [lst[2]],
                        '11': [lst[3]],
                        '12': [lst[4]],
                        '13': [lst[5]],
                        '14': [lst[6]],
                        '15': [lst[7]],
                        '16': [lst[8]],
                        '17': [lst[9]],
                        '18': [lst[10]],
                        '19': [lst[11]],
                        '20': [lst[12]],
                        '21': [lst[13]]
                        }
        df1 = DataFrame(Demand_hours, columns=['ID','Date','Weekday','8','9','10','11','12','13','14','15','16','17','18','19','20','21'])
        df = df.append(df1, ignore_index=True)
    return(df)
    
def number_shifts_id(data): #input only one store
    """ it calculates the label "Num_shift" that indicates the maximum number of shifts 
    contaiend in a working day. It returns a dataframe with the Date, this label and an ID
    made by the date.
    
        data:dataframe with the shifts information
    """
    dates=data.TransactionDate.unique()
    dates.sort()
    df=DataFrame()
    for date in dates:
        #print(date)
        date_str= str(date)
        ID=date_str[:4]+date_str[5:7]+date_str[8:10]
        subset = data[data.TransactionDate == date]
        num_shifts = subset['Shift'].count()
        #print(date)
        Shifts_count = {'ID' : [ID],
                        'Date': [date],
                        'Num_shifts': [num_shifts]
                        }
        df1 = DataFrame(Shifts_count, columns=['ID','Date','Num_shifts'])
        df = df.append(df1,ignore_index=True)
    return(df)

def auto_labelling_id(data): #input only one store
    """ it calculates the following labels, that describe the schedule by weekday and
        half an hour slot: 'Total_overlap_4','Total_overlap_3','Total_overlap_2','Starting_time_ov_4',
            'Starting_time_ov_3','Starting_time_ov_2'. It returns a dataframe with the labels for
            each day, and an ID made by the date.
    
        data:dataframe with the shifts information
    """
    shifts_subset = shifts_data[shifts_data.StoreID == 149087]
    dates=shifts_subset.TransactionDate.unique()
    dates.sort()
    df=DataFrame()
    for date in dates:
        #print(date)
        date_str= str(date)
        ID=date_str[:4]+date_str[5:7]+date_str[8:10]
        subset = data[data.TransactionDate == date]
        shifts = subset.Shift
        demand_people = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for s in shifts:
            h_i = int(s[:2])
            min_i = int(s[3:6])
            h_f = int(s[8:10])
            min_f = int(s[11:13])
            if(min_i>=30): #if the initial min is equal or higher than 30, we count it 
                #in the slot: "h.30 - h+1"
                pos_i = (h_i - 5)*2+1
            else:
                pos_i = (h_i - 5)*2
            if(min_f>=30): #if the initial min is equal or higher than 30, we count it 
                #in the slot: "h.30 - h+1"
                n = (h_f - 5)*2+1 - pos_i #total number of hours: = pos final - pos initial
            else:
                n = (h_f - 5)*2 - pos_i
            for i in range(0,n):
                demand_people[i+pos_i]+=1
        #we create a reverse copy of the vector so we can easily calculate the extension of
        #the overlaps
        demand_people_rev = demand_people.copy()
        demand_people_rev.reverse()
        #now that we have the demand_people vector, we need to extract the labels
        try:
            pos_i = demand_people.index(2) #returns the position where the value 2 is first found in the vector
            #we need to turn the position into a time: 
            ov_2_t = pos_i/2 + 5
            pos_f = len(demand_people_rev)-demand_people_rev.index(2)
            ov_2 = pos_f - pos_i
        except:
            ov_2_t = 0
            ov_2 = 0 # if no value 2 is found, then the total number of hours with 2 people overlap is 0
        try:
            pos_i = demand_people.index(3)
            ov_3_t = pos_i/2 + 5
            pos_f = len(demand_people_rev)-demand_people_rev.index(3)
            ov_3 = pos_f - pos_i
        except:
            ov_3_t = 0
            ov_3 = 0
        try:
            pos_i = demand_people.index(4)
            ov_4_t = pos_i/2 + 5
            pos_f = len(demand_people_rev)-demand_people_rev.index(4)
            ov_4 = pos_f - pos_i
        except:
            ov_4_t = 0
            ov_4 = 0
        #print(date)
        Shifts_desc = {'ID' : [ID],
                        'Date': [date],
                        'Total_overlap_4' : ov_4, #THIS COUNTS HALF HOURS
                        'Total_overlap_3' : ov_3, #THIS COUNTS HALF HOURS
                        'Total_overlap_2' : ov_2, #THIS COUNTS HALF HOURS
                        'Starting_time_ov_4' : ov_4_t,
                        'Starting_time_ov_3' : ov_3_t, 
                        'Starting_time_ov_2' : ov_2_t
                        }
        df1 = DataFrame(Shifts_desc, columns=['ID','Date','Total_overlap_4','Total_overlap_3',
                                              'Total_overlap_2','Starting_time_ov_4',
                                              'Starting_time_ov_3','Starting_time_ov_2'])
        df = df.append(df1,ignore_index=True)
    return(df)



def format_transform_for_opt(df):
    """received as an input a dataframe with the following labels:'Total_overlap_4','Total_overlap_3','Total_overlap_2','Starting_time_ov_4',
            'Starting_time_ov_3','Starting_time_ov_2'. It builds a schedule(demand in people by
            weekday and half an hour slot)in a format that the optimisation algorithm can read.
            df: dataframe that containes the labels predicted by the machine learning models
        
    """
    Common_data = {'interval': ['5-5.5','5.5-6','6-6.5','6.5-7','7-7.5','7.5-8','8-8.5','8.5-9','9-9.5','9.5-10','10-10.5','10.5-11','11-11.5','11.5-12','12-12.5','12.5-13','13-13.5','13.5-14','14-14.5','14.5-15','15-15.5','15.5-16','16-16.5','16.5-17','17-17.5','17.5-18','18-18.5','18.5-19','19-19.5','19.5-20','20-20.5','20.5-21','21-21.5','21.5-22','22-22.5','22.5-23'], 
                   'intervalIndex' : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
                   'role' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                   }
        
    df_common =  DataFrame (Common_data, columns = ['interval','intervalIndex','role'])
    #the list demand_people will be added to the df_common and will indicate the number of people
    #necessary for that specific slot
    demand_people = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #I have to start with the lowest number of overlaps until the highest, to avoid overwriting
    #the lowest
    df_final=DataFrame()
    for w in range(0,7):
        ov_2 = df[df.Weekday==w]['Total_overlap_2'].iloc[0]
        ov_3 = df[df.Weekday==w]['Total_overlap_3'].iloc[0]
        ov_4 = df[df.Weekday==w]['Total_overlap_4'].iloc[0]
        ov_2_t = df[df.Weekday==w]['Starting_time_ov_2'].iloc[0]
        ov_3_t = df[df.Weekday==w]['Starting_time_ov_3'].iloc[0]
        ov_4_t = df[df.Weekday==w]['Starting_time_ov_4'].iloc[0]
        d = demand_people.copy()
        if(ov_2>0):
            for i in range(0,int(ov_2)):
                #I am assuming that the starting hour it's never going to be "half past" but if
                #later I want to add it, I would just have to check it ov_2_t has ".5" and start
                #indexing from d[((ov_2_t-5)*2)+1 + i] instead
                d[int(ov_2_t-5)*2 + i]=2
        if(ov_3>0):
            for i in range(0,int(ov_3)):
                d[int(ov_3_t-5)*2 + i]=3
        if(ov_4>0):
            for i in range(0,int(ov_4)):
                d[int(ov_4_t-5)*2 + i]=4
        for j in range(0,len(d)):
        #I assume that the rest of the time slots, we're going to have 1 person working
        #In any case, we can filter this afterwards with the store opening times
            if(d[j]==0):
                d[j]=1
        df_weekday = df_common.copy() #we make a copy to not change the original
        df_weekday['Weekday']=[w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w] # adds the weekday
        df_weekday['demand_people']=d #adds the demand
        df_final = pd.concat([df_final,df_weekday], ignore_index = True) #adds the df of the weekday to the final df
    cols=['Weekday','interval','intervalIndex','role','demand_people'] # the order we want for the columns
    return(df_final[cols]) # the final df

def calc_position (time_reference, time):
    """calculate the position of the time "time" in a vector that starts with the time "time_reference"
        - time_reference: time that is placed in the first element of the vector
        - time: position of the time we want to calculate
    """
    pos = time - time_reference
    if(pos - int(pos) != 0):
        pos = int(pos)*2+1
    else:
        pos *=2
    return pos

def adapting_to_openingtimes(df,time_reference,mon_op,mon_cl,tue_op,tue_cl,wed_op,wed_cl,thu_op,thu_cl,fri_op,fri_cl,sat_op,sat_cl,sun_op,sun_cl):
    """adapts the final schedule calculated by using the predictions to the opening times of the store
        - df: position of the time we want to calculate
    """
    times = [wed_op,wed_cl,thu_op,thu_cl,fri_op,fri_cl,sat_op,sat_cl,sun_op,sun_cl,mon_op,mon_cl,tue_op,tue_cl]
    weekdays = [0,1,2,3,4,5,6]
    for w in weekdays:
        #goes through opening times
        df.loc[(df.Weekday == w) &(df.intervalIndex<calc_position(8,times[w*2])),'demand_people']=0
    for w in weekdays:
        #goes through closing times
        df.loc[(df.Weekday == w) &(df.intervalIndex>=calc_position(8,times[w*2+1])),'demand_people']=0
    return(df)


################################################################################################
#......................................Main code...........................

fileDir = os.path.dirname(os.path.abspath(__file__))#to get the file path


################################# PRE - PROCESSING ######################################
###################............. DEMAND DATA.......................
print("Reading and pre - processing the demand data...")

if(DEMAND_USING_FORMULA):
    #read the csv and set the columns' names
    colnames=['OrganisationID', 'StoreInstanceID', 'StoreID', 'Date', 'Hour', 'sales', 'units']
    sales_data1 = pd.read_csv(os.path.join(fileDir, 'Data/'+SALES_FILE),
                             names=colnames, header=None)
    sales_data2 = pd.read_csv(os.path.join(fileDir, 'Data/'+SALES_FILE2),
                              names=colnames, header=None)
    
    #formatting
    sales_data1['Date']=sales_data1['Date'].apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d'))#change to date format
    sales_data2['Date']=sales_data2['Date'].apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d'))#change to date format
    sales_data = pd.concat([sales_data1,sales_data2], ignore_index=True)#concat the two datasets, since they have the same data but different dates

    #Calculate the NewDate. The real demand from one year will be assigned as forecasted demand for next
    #year "NewDate"
    print("Calculating next year demand...")
    sales_data['NewDate'] = sales_data['Date'].apply(lambda x: get_next_year_demand(x.weekofyear,(x.weekday()+1),x.year))
    #weekday + 1 is because this variable is Timestamp type, so Monday=0, Sunday=6. However, inside the function
        #the function used to calculate the next year date uses Monday=1, Sunday=0. So we pass as an argument
        #weekday+1 and internally in the function we change if weekday=7 --> weekday=0 (which is sunday)
else:
    colnames=['OrganisationID', 'StoreInstanceID', 'StoreID', 'Date', 'Hour', 'sales', 'units']
    sales_data = pd.read_csv(os.path.join(fileDir, 'Data/'+FORECASTED_DEMAND_FILE),
                             names=colnames, header=None)
    sales_data['Date']=sales_data['Date'].apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d'))#change to date format
    sales_data['NewDate'] = sales_data['Date']
###################................SHIFTS DATA..........................
print("Reading and pre - processing shifts data...")
#read the files
shifts_data = pd.read_csv(os.path.join(fileDir, 'Data/'+SHIFTS_FILE))
shifts_time = pd.read_csv(os.path.join(fileDir, 'Data/'+SHIFTS_FILE_TIMES))
shifts_data.StartTime = shifts_time.StartTime
shifts_data.FinishTime = shifts_time.FinishTime

#formatting
shifts_data['TransactionDate'] = shifts_data['TransactionDate'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
shifts_data['StartTime'] = shifts_data['StartTime'].apply(lambda x: datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S'))#change to date format
shifts_data['FinishTime'] = shifts_data['FinishTime'].apply(lambda x: datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S'))#change to date format
shifts_data['Shift'] = shifts_data['StartTime'].apply(lambda x: x.strftime("%H:%M"))+" - "+shifts_data['FinishTime'].apply(lambda x: x.strftime("%H:%M"))
shifts_data['Hour_start'] = shifts_data['StartTime'].apply(lambda x: int(x.strftime("%H")))
shifts_data['Hour_finish'] = shifts_data['FinishTime'].apply(lambda x: int(x.strftime("%H")))
#If we consider outliers as the values which are three times the standars deviation from the mean,
#we can say that the following values are outliers:
low_thershole_WP = np.mean(shifts_data.WorkPeriod) - np.std(shifts_data.WorkPeriod)*3
high_thershole_WP = np.mean(shifts_data.WorkPeriod) + np.std(shifts_data.WorkPeriod)*3
outlier_WP = (shifts_data['WorkPeriod']>high_thershole_WP) | (shifts_data['WorkPeriod']<low_thershole_WP)
#I delete the outliers
shifts_data = shifts_data[~ outlier_WP]
#Although, I still have really strage values for this variable, like 0.25 0.5 or 1... I delete the workperiod shorter than 2h:
WP_shorterThan2h = (shifts_data['WorkPeriod']<2)
shifts_data=shifts_data[~ WP_shorterThan2h]

#workers and demand unified
sales_data['NumberofWorkers'] = sales_data.apply(lambda row: get_number_employees_store(row.NewDate,row.Hour,shifts_data,row.StoreID),axis=1)
sales_data.to_csv("Data/sales_workers_data2.csv", index=False)
sales_workers_data = pd.read_csv(os.path.join(fileDir, 'Data/sales_workers_data2.csv'))
sales_workers_data['Month']=sales_workers_data.NewDate.apply(lambda x: get_month(x))
sales_workers_data['Weekday']=sales_workers_data.NewDate.apply(lambda x: get_weekday(x))
#.......Shifts: data pre- processing.........
#check for general duplicates
shifts_data[shifts_data.duplicated()]
#There is no rows that are exactly the same

#check for duplicates employees and days
shifts_data[shifts_data.duplicated(['EmployeeID', 'StartTime','FinishTime'])]
#delete the duplicates
shifts_data.drop_duplicates(subset =['EmployeeID', 'StartTime','FinishTime'], keep = 'first', inplace = True) 
#Keep = ‘first’, it considers first value as unique and rest of the same values as duplicate.
#missing values
shifts_data.apply(lambda x: sum(x.isna()))
#Only the EmployeeID present missing values, and in this case for these models we don't care
#about this field becasue we don't care about the identity of the worker, so we keep it


###################................DATASET CREATION: ATTRIBUTES AND LABELLING..........................
print("Creating the dataset...")
print("Adding the features...")

subset = sales_workers_data[(sales_workers_data.StoreID == STOREID)]
features_store = features_demand_id(subset)
shifts_subset = shifts_data[shifts_data.StoreID == STOREID]
shifts_store = number_shifts_id(shifts_subset)

print("Automatic labelling...")
shifts_store2 = auto_labelling_id(shifts_subset)

merge = pd.merge(features_store, shifts_store, on='ID')
merge = pd.merge(merge, shifts_store2, on='ID')
merge = merge.drop("Date_y", axis=1)
merge = merge.drop("ID", axis=1)
merge = merge.drop("Date", axis=1)
merge.to_csv('DT_datasets/dataset_'+str(STOREID)+'.csv', index=False)


#################### PREDICTIONS USING THE PARAMETERS OBTAINED ##################
#######################      Number of shifts    ###########################################
ds = pd.read_csv(os.path.join(fileDir, 'DT_datasets/dataset_'+str(STOREID)+'.csv'), sep = ',')

#....................... Last preprocessing of the data ....................
#checking missing values:
ds.apply(lambda x: sum(x.isna()))
ds[ds['8'].isna()]['Weekday'].value_counts()
ds[ds['21'].isna()]['Weekday'].value_counts()
#The opening hours for this store are: Mon - Sat: 7h-21h and Sun: 8h-20h so it makes sense that for Sundays we have a lot
#of missing values

#For missing values for 8h and 21h for sundays and 8h for Saturdays I replace them for 0, because the most likely
#reason is that there's no demand:
mask = (ds['8'].isna()) & ((ds.Weekday =='Saturday') |(ds.Weekday =='Sunday'))
ds.loc[mask, '8'] = 0 #set the values to zero
mask2 = (ds['21'].isna()) & (ds.Weekday =='Sunday')
ds.loc[mask2, '21'] = 0

#For the rest of missing values, maybe it's better to delete the whole row as there are not a lot of them and we don't know
#for sure if there was no demand or if there was demand but the value is missing. For the moment we are goint to set all of them to zeros:
ds.loc[ds['8'].isna(), '8'] = 0
ds.loc[ds['9'].isna(), '9'] = 0
ds.loc[ds['10'].isna(), '10'] = 0
ds.loc[ds['11'].isna(), '11'] = 0
ds.loc[ds['12'].isna(), '12'] = 0
ds.loc[ds['13'].isna(), '13'] = 0
ds.loc[ds['14'].isna(), '14'] = 0
ds.loc[ds['15'].isna(), '15'] = 0
ds.loc[ds['16'].isna(), '16'] = 0
ds.loc[ds['17'].isna(), '17'] = 0
ds.loc[ds['18'].isna(), '18'] = 0
ds.loc[ds['19'].isna(), '19'] = 0
ds.loc[ds['20'].isna(), '20'] = 0
ds.loc[ds['21'].isna(), '21'] = 0
#check that there are no missing values left
ds.apply(lambda x: sum(x.isna()))

#transform categorical attributes
ds.Weekday.value_counts()
#IMPORTANT: NUMBERS NOW ARE THE SAME FORMAT THAT THE INPUT OF THE OPTIMISATION ALGORITHM(Wed=0, Tue=6)
categorical_to_num = {"Weekday":     {"Monday": 5, "Tuesday": 6, "Wednesday": 0, "Thursday": 1, "Friday": 2, "Saturday": 3, "Sunday": 4}}
ds.replace(categorical_to_num, inplace=True)
ds['Weekday']=pd.to_numeric(ds.Weekday)



#here data is separated into training and test datasets. We leave out of the training test(which we will
#use for parameter tunning)
mask_weeks_out = (~ds['Date_x'].isin(W33))&(~ds['Date_x'].isin(W4))&(~ds['Date_x'].isin(W15))&(~ds['Date_x'].isin(W46))
ds_training=ds[mask_weeks_out]
ds_test=ds[~mask_weeks_out]

#deleting the date from the dataset
ds_training = ds_training.drop(['Date_x'], axis=1)
features_149087 = ["Weekday","8","9","10","11","12","13","14","15","16","17","18","19","20","21"]
labels_149087 = ["Total_overlap_4","Total_overlap_3","Total_overlap_2","Starting_time_ov_4","Starting_time_ov_3","Starting_time_ov_2"]


X_train = ds_training[features_149087]
Y_train = ds_training["Num_shifts"]

X_test = ds_test[features_149087]
Y_test = ds_test["Num_shifts"]


#-----------------          DT regressor          ---------------------------------
tree_reg = DecisionTreeRegressor(criterion ='mse' ,splitter='random', max_depth=3, min_samples_split=20,min_samples_leaf=16,max_features='auto')
model_reg = tree_reg.fit(X_train, Y_train) #training
X_test_reg = ds_test[features_149087]
y_pred_DTreg = tree_reg.predict(X_test_reg)
print("RMSE for Num_shifts:")
root_mean_squared_error(Y_test, y_pred_DTreg)
X_test['DT_reg_pred'] = np.round(y_pred_DTreg, 0)
X_test.to_csv('Results/Predictions/Number_shifts_'+str(STOREID)+'.csv', index = False)




#######################      Demand in people    ###########################################
#-----------------          DT classifier          ---------------------------------
#------- Total_overlap_4 -------
tree_class_TO4 = DecisionTreeClassifier(criterion = 'entropy' ,splitter= 'random', max_depth=8, min_samples_split=19, min_samples_leaf=10,max_features=None)

X_train_TO4 = ds_training[features_149087]
Y_train_TO4 = ds_training["Total_overlap_4"]

X_test_TO4 = ds_test[features_149087]
Y_test_TO4 = ds_test["Total_overlap_4"]

model = tree_class_TO4.fit(X_train_TO4, Y_train_TO4) #training
y_pred_DTclass_TO4 = tree_class_TO4.predict(X_test_TO4)
print("Accuracy for Total_overlap_4:")
accuracy_score(Y_test_TO4, y_pred_DTclass_TO4)
print("Precision for Total_overlap_4:")
precision_score(Y_test_TO4, y_pred_DTclass_TO4, average="weighted") #labels=np.unique(y_pred) can be added to calculate the measure only for the labels that have predicted samples
print("Recall for Total_overlap_4:")
recall_score(Y_test_TO4, y_pred_DTclass_TO4, average="weighted")
print("F1-Score for Total_overlap_4:")
f1_score(Y_test_TO4, y_pred_DTclass_TO4, average="weighted")

X_test_TO4['Date']=ds_test['Date_x']
X_test_TO4['TO4_class_pred'] = y_pred_DTclass_TO4.copy()

#------- Total_overlap_3 -------
tree_class_TO3 = DecisionTreeClassifier(criterion = 'gini' ,splitter= 'best', max_depth=5, min_samples_split=22, min_samples_leaf=3,max_features=None)

X_train_TO3 = ds_training[features_149087]
Y_train_TO3 = ds_training["Total_overlap_3"]

X_test_TO3 = ds_test[features_149087]
Y_test_TO3 = ds_test["Total_overlap_3"]

model = tree_class_TO3.fit(X_train_TO3, Y_train_TO3) #training
y_pred_DTclass_TO3 = tree_class_TO3.predict(X_test_TO3)
print("Accuracy for Total_overlap_3:")
accuracy_score(Y_test_TO3, y_pred_DTclass_TO3)
print("Precision for Total_overlap_3:")
precision_score(Y_test_TO3, y_pred_DTclass_TO3, average="weighted") #labels=np.unique(y_pred) can be added to calculate the measure only for the labels that have predicted samples
print("Recall for Total_overlap_3:")
recall_score(Y_test_TO3, y_pred_DTclass_TO3, average="weighted")
print("F1-Score for Total_overlap_3:")
f1_score(Y_test_TO3, y_pred_DTclass_TO3, average="weighted")
X_test_TO4['TO3_class_pred'] = y_pred_DTclass_TO3.copy()

#------- Total_overlap_2 -------
tree_class_TO2 = DecisionTreeClassifier(criterion = 'entropy' ,splitter= 'random', max_depth=9, min_samples_split=16, min_samples_leaf=20,max_features=None)

X_train_TO2 = ds_training[features_149087]
Y_train_TO2 = ds_training["Total_overlap_2"]

X_test_TO2 = ds_test[features_149087]
Y_test_TO2 = ds_test["Total_overlap_2"]

model = tree_class_TO2.fit(X_train_TO2, Y_train_TO2) #training
y_pred_DTclass_TO2 = tree_class_TO2.predict(X_test_TO2)
print("Accuracy for Total_overlap_2:")
accuracy_score(Y_test_TO2, y_pred_DTclass_TO2)
print("Precision for Total_overlap_2:")
precision_score(Y_test_TO2, y_pred_DTclass_TO2, average="weighted") #labels=np.unique(y_pred) can be added to calculate the measure only for the labels that have predicted samples
print("Recall for Total_overlap_2:")
recall_score(Y_test_TO2, y_pred_DTclass_TO2, average="weighted")
print("F1-Score for Total_overlap_2:")
f1_score(Y_test_TO2, y_pred_DTclass_TO2, average="weighted")

X_test_TO4['TO2_class_pred'] = y_pred_DTclass_TO2.copy()


#------- Starting_time_ov_4 -------
tree_class_ST4 = DecisionTreeClassifier(criterion = 'entropy' ,splitter= 'random', max_depth=7, min_samples_split=11, min_samples_leaf=7,max_features=None)

X_train_ST4 = ds_training[features_149087]
Y_train_ST4 = ds_training["Starting_time_ov_4"]

X_test_ST4 = ds_test[features_149087]
Y_test_ST4 = ds_test["Starting_time_ov_4"]

model = tree_class_ST4.fit(X_train_ST4, Y_train_ST4) #training

y_pred_DTclass_ST4 = tree_class_ST4.predict(X_test_ST4)
print("Accuracy for Starting_time_ov_4:")
accuracy_score(Y_test_ST4, y_pred_DTclass_ST4)
print("Precision for Starting_time_ov_4:")
precision_score(Y_test_ST4, y_pred_DTclass_ST4, average="weighted") #labels=np.unique(y_pred) can be added to calculate the measure only for the labels that have predicted samples
print("Recall for Starting_time_ov_4:")
recall_score(Y_test_ST4, y_pred_DTclass_ST4, average="weighted")
print("F1-Score for Starting_time_ov_4:")
f1_score(Y_test_ST4, y_pred_DTclass_ST4, average="weighted")
X_test_TO4['ST4_class_pred'] = y_pred_DTclass_ST4.copy()

#-----------------          DT regressor          ---------------------------------

#------- Starting_time_ov_3 -------
tree_reg_ST3 = DecisionTreeRegressor(criterion ='mse' ,splitter='best', max_depth=9, min_samples_split=14,min_samples_leaf=3,max_features='auto')

X_train_ST3 = ds_training[features_149087]
Y_train_ST3 = ds_training["Starting_time_ov_3"]

X_test_ST3 = ds_test[features_149087]
Y_test_ST3 = ds_test["Starting_time_ov_3"]

model_reg = tree_reg_ST3.fit(X_train_ST3, Y_train_ST3) #training
X_test_reg_ST3 = ds_test[features_149087]
y_pred_DTreg_ST3 = tree_reg_ST3.predict(X_test_reg_ST3)
print("RMSE for Starting_time_ov_3:")
root_mean_squared_error(Y_test_ST3, y_pred_DTreg_ST3)
X_test_TO4['ST3_reg_pred'] = y_pred_DTreg_ST3.copy()

#------- Starting_time_ov_2 -------
tree_reg_ST2 = DecisionTreeRegressor(criterion ='mae' ,splitter='random', max_depth=None, min_samples_split=30,min_samples_leaf=7,max_features=None)

X_train_ST2 = ds_training[features_149087]
Y_train_ST2 = ds_training["Starting_time_ov_2"]

X_test_ST2 = ds_test[features_149087]
Y_test_ST2 = ds_test["Starting_time_ov_2"]

model_reg = tree_reg_ST2.fit(X_train_ST2, Y_train_ST2) #training
X_test_reg_ST2 = ds_test[features_149087]
y_pred_DTreg_ST2 = tree_reg_ST2.predict(X_test_reg_ST2)
print("RMSE for Starting_time_ov_2:")
root_mean_squared_error(Y_test_ST2, y_pred_DTreg_ST2)
X_test_TO4['ST2_reg_pred'] = y_pred_DTreg_ST2.copy()
X_test_TO4.to_csv('Results/Predictions/Demand_in_people'+str(STOREID)+'.csv', index = False)




###################................CONVERTING PREDICTIONS INTO SCHEDULES..........................
pos = (STARTING_TIME - 5)*2
#...................Main code for DT predictions: ...............
schedule_data = pd.read_csv(os.path.join(fileDir, 'Results/Predictions/Demand_in_people'+str(STOREID)+'.csv'))

schedule_data_W1 = schedule_data[schedule_data['Date'].isin(W33)]
schedule_data_W2 = schedule_data[schedule_data['Date'].isin(W46)]
schedule_data_W3 = schedule_data[schedule_data['Date'].isin(W4)]
schedule_data_W4 = schedule_data[schedule_data['Date'].isin(W15)]


###......................... Classification results.............
###Dataframe for W1
schedule_data_class_W1 = DataFrame()
schedule_data_class_W1['Weekday']=schedule_data_W1['Weekday']
schedule_data_class_W1['Total_overlap_4']=schedule_data_W1['TO4_class_pred']
schedule_data_class_W1['Total_overlap_3']=schedule_data_W1['TO3_class_pred']
schedule_data_class_W1['Total_overlap_2']=schedule_data_W1['TO2_class_pred']
schedule_data_class_W1['Starting_time_ov_4']=schedule_data_W1['ST4_class_pred']
schedule_data_class_W1['Starting_time_ov_3'] = np.round(schedule_data_W1['ST3_reg_pred'], 0)
schedule_data_class_W1['Starting_time_ov_2']= np.round(schedule_data_W1['ST2_reg_pred'], 0)

demand_in_people_class_W1 = format_transform_for_opt(schedule_data_class_W1)
#Results using the classification DT predictions for Total_overlap_4,Total_overlap_3,Total_overlap_2
# and Starting_time_ov_4 and the regression for Starting_time_ov_3 and Starting_time_ov_2
#demand_in_people_class_W1.to_csv("Results/Demand_in_people/DT_class_schedule_150818.csv", index = False)
demand_in_people_class_W1_8am = demand_in_people_class_W1[demand_in_people_class_W1['intervalIndex']>=pos]#filter from 8am(the algorithm is not design yet to start from 5am)
demand_in_people_class_W1_8am['intervalIndex'] = demand_in_people_class_W1_8am.intervalIndex-pos #intervalIndex adapted to start at 8am
demand_in_people_class_W1_8am= adapting_to_openingtimes(demand_in_people_class_W1_8am,8,8,21,8,21,8,21,8,21,8,21,8,21,8,20)
demand_in_people_class_W1_8am.to_csv('Results/Predictions/Schedule(DEMAND)_'+str(STOREID)+'W1.csv', index = False)

###Dataframe for W2
schedule_data_class_W2 = DataFrame()
schedule_data_class_W2['Weekday']=schedule_data_W2['Weekday']
schedule_data_class_W2['Total_overlap_4']=schedule_data_W2['TO4_class_pred']
schedule_data_class_W2['Total_overlap_3']=schedule_data_W2['TO3_class_pred']
schedule_data_class_W2['Total_overlap_2']=schedule_data_W2['TO2_class_pred']
schedule_data_class_W2['Starting_time_ov_4']=schedule_data_W2['ST4_class_pred']
schedule_data_class_W2['Starting_time_ov_3'] = np.round(schedule_data_W2['ST3_reg_pred'], 0)
schedule_data_class_W2['Starting_time_ov_2']= np.round(schedule_data_W2['ST2_reg_pred'], 0)

demand_in_people_class_W2 = format_transform_for_opt(schedule_data_class_W2)
#Results using the classification DT predictions for Total_overlap_4,Total_overlap_3,Total_overlap_2
# and Starting_time_ov_4 and the regression for Starting_time_ov_3 and Starting_time_ov_2
demand_in_people_class_W2_8am = demand_in_people_class_W2[demand_in_people_class_W2['intervalIndex']>=pos]
demand_in_people_class_W2_8am['intervalIndex'] = demand_in_people_class_W2_8am.intervalIndex-pos
demand_in_people_class_W2_8am= adapting_to_openingtimes(demand_in_people_class_W2_8am,8,8,21,8,21,8,21,8,21,8,21,8,21,8,20)
demand_in_people_class_W2_8am.to_csv('Results/Predictions/Schedule(DEMAND)_'+str(STOREID)+'W2.csv', index = False)

###Dataframe for W3
schedule_data_class_W3 = DataFrame()
schedule_data_class_W3['Weekday']=schedule_data_W3['Weekday']
schedule_data_class_W3['Total_overlap_4']=schedule_data_W3['TO4_class_pred']
schedule_data_class_W3['Total_overlap_3']=schedule_data_W3['TO3_class_pred']
schedule_data_class_W3['Total_overlap_2']=schedule_data_W3['TO2_class_pred']
schedule_data_class_W3['Starting_time_ov_4']=schedule_data_W3['ST4_class_pred']
schedule_data_class_W3['Starting_time_ov_3'] = np.round(schedule_data_W3['ST3_reg_pred'], 0)
schedule_data_class_W3['Starting_time_ov_2']= np.round(schedule_data_W3['ST2_reg_pred'], 0)

demand_in_people_class_W3 = format_transform_for_opt(schedule_data_class_W3)
#Results using the classification DT predictions for Total_overlap_4,Total_overlap_3,Total_overlap_2
# and Starting_time_ov_4 and the regression for Starting_time_ov_3 and Starting_time_ov_2
demand_in_people_class_W3_8am = demand_in_people_class_W3[demand_in_people_class_W3['intervalIndex']>=pos]
demand_in_people_class_W3_8am['intervalIndex'] = demand_in_people_class_W3_8am.intervalIndex-pos
demand_in_people_class_W3_8am= adapting_to_openingtimes(demand_in_people_class_W3_8am,8,8,21,8,21,8,21,8,21,8,21,8,21,8,20)
demand_in_people_class_W3_8am.to_csv('Results/Predictions/Schedule(DEMAND)_'+str(STOREID)+'W3.csv', index = False)

###Dataframe for W1
schedule_data_class_W4 = DataFrame()
schedule_data_class_W4['Weekday']=schedule_data_W4['Weekday']
schedule_data_class_W4['Total_overlap_4']=schedule_data_W4['TO4_class_pred']
schedule_data_class_W4['Total_overlap_3']=schedule_data_W4['TO3_class_pred']
schedule_data_class_W4['Total_overlap_2']=schedule_data_W4['TO2_class_pred']
schedule_data_class_W4['Starting_time_ov_4']=schedule_data_W4['ST4_class_pred']
schedule_data_class_W4['Starting_time_ov_3'] = np.round(schedule_data_W4['ST3_reg_pred'], 0)
schedule_data_class_W4['Starting_time_ov_2']= np.round(schedule_data_W4['ST2_reg_pred'], 0)

demand_in_people_class_W4 = format_transform_for_opt(schedule_data_class_W4)
#Results using the classification DT predictions for Total_overlap_4,Total_overlap_3,Total_overlap_2
# and Starting_time_ov_4 and the regression for Starting_time_ov_3 and Starting_time_ov_2
demand_in_people_class_W4_8am = demand_in_people_class_W4[demand_in_people_class_W4['intervalIndex']>=pos]
demand_in_people_class_W4_8am['intervalIndex'] = demand_in_people_class_W4_8am.intervalIndex-pos
demand_in_people_class_W4_8am= adapting_to_openingtimes(demand_in_people_class_W4_8am,8,8,21,8,21,8,21,8,21,8,21,8,21,8,20)
demand_in_people_class_W4_8am.to_csv('Results/Predictions/Schedule(DEMAND)_'+str(STOREID)+'W4.csv', index = False)





