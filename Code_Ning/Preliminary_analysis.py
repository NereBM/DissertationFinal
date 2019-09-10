# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import sys
import os
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from time import mktime
import time


fileDir = os.path.dirname(os.path.abspath(__file__))#to get the file path

####---------------------Auxiliar functions ---------------------------------------


def get_next_year_demand(weeknum,weekday,year):
    #WATCH OUT THE DAY OF THE WEEK!! for Timestamp Monday=0, Sunday=6
    if(year==2018):
        weeknum -=1#if we calculate the 2019 date we need to substract 1 to the week number
                   #because it counts one week ahead from calendar for this year
    if(weekday==7):
        weekday=0
    day = time.asctime(time.strptime(str(year+1)+' '+str(weeknum)+' '+str(weekday), '%Y %W %w'))#Monday=1, Sunday=0
    return(datetime.strptime(day, "%a %b %d %H:%M:%S %Y"))

#PAY ATTENTION WHEN PASSING THE DAY OF THE WEEK!! for Timestamp Monday=0, Sunday=6


def get_number_employees(date,hour,df):
    return(len(df[(df.TransactionDate == date) & (hour>=df.Hour_start) & (hour<df.Hour_finish)]))

def get_number_employees_store(date,hour,df,store):
    return(len(df[(df.TransactionDate == date) & (hour>=df.Hour_start) & (hour<df.Hour_finish) & (df.StoreID==store)]))
  
def get_month(date):
    date_format=datetime.strptime(date, "%Y-%m-%d")
    return (date_format.month)

def get_weekday(date):
    date_format=datetime.strptime(date, "%Y-%m-%d")
    wd=date_format.weekday()
    #<expression1> if <condition> else <expression2>
    return("Monday" if wd==0 else ("Tuesday" if wd==1 else
             ("Wednesday" if wd==2 else 
                  ("Thursday" if wd==3 else 
                       ("Friday" if wd==4 else
                            ("Saturday" if wd==5 else "Sunday"))))))

def scatterplots(storeid, sales_workers_dat):
    sw_store = sales_workers_dat[(sales_workers_dat["StoreID"]==storeid) & (sales_workers_dat["NumberofWorkers"]!=0)]
   
    
    #scatterplot sales vs number of workers of store 149091 grouped by month
    g = sns.FacetGrid(sw_store, col="Month", col_wrap=3, height=3.5)
    g= g.map_dataframe(plt.scatter, "NumberofWorkers", "sales")
    g.savefig("corr_month"+str(storeid)+".png")
    
    #scatterplot sales vs number of workers of store 149091 grouped by weekday
    g2 = sns.FacetGrid(sw_store, col="Weekday", col_wrap=3, height=3.5)
    g2.map_dataframe(plt.scatter, "NumberofWorkers", "sales")
    g2.savefig("corr_weekday"+str(storeid)+".png")
    
    #scatterplot sales vs number of workers of store 149091 grouped by store instance
    g2 = sns.FacetGrid(sw_store, col="StoreInstanceID", col_wrap=3, height=3.5)
    g2.map_dataframe(plt.scatter, "NumberofWorkers", "sales")
    g2.savefig("corr_manager"+str(storeid)+".png")
    
    #general scatterplot
    plt.scatter(sw_store.NumberofWorkers, sw_store.sales, c=(0,0,0), alpha=0.5)



########################################################################
##########------- READING FROM FILES AND DATA FORMATTING ------#########
    ########################################################################
employee_constraints = pd.read_csv(os.path.join(fileDir, 'Data/EmployeeScheduleConstraints.csv'))
employee_unavailable = pd.read_csv(os.path.join(fileDir, 'Data/EmployeeUnavailableHours.csv'))

#read the csv and set the columns' names
colnames=['OrganisationID', 'StoreInstanceID', 'StoreID', 'Date', 'Hour', 'sales', 'units']
sales_data1 = pd.read_csv(os.path.join(fileDir, 'Data/SalesData_Orgs_329970_339259_763128_766174.csv'),
                         names=colnames, header=None)
sales_data2 = pd.read_csv(os.path.join(fileDir, 'Data/SalesData_Orgs_780237_780251_781676_791175.csv'),
                          names=colnames, header=None)
sales_data1['Date']=sales_data1['Date'].apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d'))#change to date format
sales_data2['Date']=sales_data2['Date'].apply(lambda x: datetime.strptime(x[:10], '%Y-%m-%d'))#change to date format
sales_data = pd.concat([sales_data1,sales_data2], ignore_index=True)#concat the two datasets, since they have the same data but different dates
sales_data['NewDate'] = sales_data['Date'].apply(lambda x: get_next_year_demand(x.weekofyear,(x.weekday()+1),x.year))
forecast_data = pd.read_csv(os.path.join(fileDir, 'Data/Schedules_2018-04_2019-05.csv'))
forecast_data['ScheduleStartDate'] = forecast_data['ScheduleStartDate'].apply(lambda x: datetime.strptime(x[:10], '%d/%m/%Y'))
forecast_data['ScheduleEndingDate'] = forecast_data['ScheduleEndingDate'].apply(lambda x: datetime.strptime(x[:10], '%d/%m/%Y'))

shifts_data = pd.read_csv(os.path.join(fileDir, 'Data/ScheduleShifts_2018-04_2019-05.csv'))
shifts_time = pd.read_csv(os.path.join(fileDir, 'Data/ScheduleShifts_2018-04_2019-05_times.csv'))
shifts_data.StartTime = shifts_time.StartTime
shifts_data.FinishTime = shifts_time.FinishTime
shifts_data['TransactionDate'] = shifts_data['TransactionDate'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
shifts_data['StartTime'] = shifts_data['StartTime'].apply(lambda x: datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S'))#change to date format
shifts_data['FinishTime'] = shifts_data['FinishTime'].apply(lambda x: datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S'))#change to date format
shifts_data['Shift'] = shifts_data['StartTime'].apply(lambda x: x.strftime("%H:%M"))+" - "+shifts_data['FinishTime'].apply(lambda x: x.strftime("%H:%M"))
shifts_data['Hour_start'] = shifts_data['StartTime'].apply(lambda x: int(x.strftime("%H")))
shifts_data['Hour_finish'] = shifts_data['FinishTime'].apply(lambda x: int(x.strftime("%H")))

#workers and demand unified
#The following line takes A LONG while, my recommendation is to run it once, and save it
#in a csv, and read that csv the next times. 
#UNCOMMENT FOR THE FIRST TIME. COMMENT FOR THE FOLLOWING TIMES:
#sales_data['NumberofWorkers'] = sales_data.apply(lambda row: get_number_employees_store(row.NewDate,row.Hour,shifts_data,row.StoreID),axis=1)
#UNCOMMENT FOR THE FIRST TIME. COMMENT FOR THE FOLLOWING TIMES:(To save it in a csv)
#sales_data.to_csv('Data/sales_workers_data.csv')
#To read it from the csv, uncomment following line:
sales_workers_data = pd.read_csv(os.path.join(fileDir, 'Data/sales_workers_data.csv'))
#sales_workers_data = pd.read_csv(os.path.join(fileDir, 'Data/sales_workers_data.csv'))

########################################################################
##########----------- DATA PREPROCESSING: SHIFTS ----------#########
########################################################################
#--------------------- duplicates --------------------
#check for general duplicates
shifts_data[shifts_data.duplicated()]
#There is no rows that are exactly the same

#check for duplicates employees and days
dup = shifts_data[shifts_data.duplicated(['EmployeeID', 'StartTime','FinishTime'])]
#There is 79 duplicate values, in 67 of them, what changes is the scheduleID:
dup2 = shifts_data[shifts_data.duplicated(['EmployeeID', 'StartTime','FinishTime','ScheduleID'])]
dup2.loc[28468,['OrganisationID','StoreInstanceID','StoreID','EmployeeID','StartTime']]
dup2.EmployeeID
#The 12 repeated values, they have different ScheduleShiftID, and all the EmployeeID are missing values
shifts_data[shifts_data.duplicated(['EmployeeID', 'StartTime','FinishTime','ScheduleID','ScheduledShiftID'])]
#No repeated rows at all if we also include the ScheduledShiftID
#However, if we only compare the Scheduled ShiftID we can see that there are 32 repeated values,
#that are the same and only the ScheduleID changes. I will delete this values because for a 
#same EmployeeID and a same StoreID they have the same StartTime and FinishTime so they are being
#count twice.
dup3=shifts_data[shifts_data.duplicated(['ScheduledShiftID'])]
dup3.ScheduledShiftID

#We delete all the duplicates from the first result
shifts_data.drop_duplicates(subset =['EmployeeID', 'StartTime','FinishTime'], keep = 'first', inplace = True) 
#Keep = ‘first’, it considers first value as unique and rest of the same values as duplicate.


#--------------------- missing values --------------------
shifts_data.apply(lambda x: sum(x.isna()))
#Only the EmployeeID present missing values
#NA values dor employee ID
sum(shifts_data.EmployeeID.isna())
#I could ignore them
#Percentage of NA:
sum(shifts_data.EmployeeID.isna())/len(shifts_data.EmployeeID)*100
#As the percentage is not significant and the rest of the fields are not null, I am going to use this values in the general statistics
#for all the analysis that are not related to the EmployeeID


########################################################################
##########----------- PRELIMINARY ANALYSIS ----------#########
########################################################################


#----------------------------------------------------------------------
#---------------------- important measures, ranges and outliers ----------------
#----------------------------------------------------------------------
#For all the IDs(OrganisationID, StoreInstanceID, StoreID, ScheduleID, ScheduledShiftID, EmployeeID) 
#it does not make sense to extract this measures
#For the dates, we need to know the range:
shifts_data.TransactionDate.min()
shifts_data.StartTime.min()
shifts_data.FinishTime.min()
shifts_data.TransactionDate.max()
shifts_data.StartTime.max()
shifts_data.FinishTime.max()


#For WorkPeriod:
plt.figure(1)
plt.title('WorkPeriod')
plt.ylabel('Hours')
plt.boxplot(shifts_data.WorkPeriod)
fig1=plt.gcf()
fig1.savefig("WP_boxplot_withOutliers.png")

#If we consider outliers as the values which are three times the standars deviation from the mean,
#we can say that the following values are outliers:
low_thershole_WP = np.mean(shifts_data.WorkPeriod) - np.std(shifts_data.WorkPeriod)*3
high_thershole_WP = np.mean(shifts_data.WorkPeriod) + np.std(shifts_data.WorkPeriod)*3
outlier_WP = (shifts_data['WorkPeriod']>high_thershole_WP) | (shifts_data['WorkPeriod']<low_thershole_WP)
#I delete the outliers
shifts_data = shifts_data[~ outlier_WP]
plt.figure(2)
plt.title('WorkPeriod')
plt.ylabel('Hours')
plt.boxplot(shifts_data.WorkPeriod)
fig2=plt.gcf()
fig2.savefig("WP_boxplot_noOutliers.png")
#Although, I still have really strage values for this variable, like 0.25 0.5 or 1... I delete the workperiod shorter than 2h:
WP_shorterThan2h = (shifts_data['WorkPeriod']<2)
shifts_data=shifts_data[~ WP_shorterThan2h]
#For the boolean attributes:
shifts_data.FinishAtStoreClose.value_counts()
shifts_data.StockTake.value_counts()
shifts_data.InAttendance.value_counts()
shifts_data.OpeningShift.value_counts()
shifts_data.Off.value_counts()
shifts_data.OnHoliday.value_counts()

#how many store are included and how many registers of each one
shifts_data.StoreID.value_counts()

#Frequency of the shifts by store
shifts_freq_bystore = shifts_data.groupby(['StoreID','Shift'])
shifts_freq_bystore.Shift.count()

#----------------------------------------------------------------------
#---------------------- shift histograms ----------------
#----------------------------------------------------------------------

#Data for important stores, ordered by number of rows: 
############ Store 146633 - Pride Park: 2383 samples
store_146633 = shifts_data[shifts_data["StoreID"]==146633]
len(store_146633)
store_146633.Shift.value_counts()
store_146633.Shift.value_counts()[:30].plot(kind='bar', title="Shifts histogram: store 146633")
plt.savefig("histogram_146633.png", bbox_inches="tight")
fig3=plt.gcf()
#resize for report
fig3.set_size_inches(15, 11)
fig3.savefig("histogram_146633_1.png")

############ Store 396666 - Derby, Kingsway: 1619 samples
store_396666 = shifts_data[shifts_data["StoreID"]==396666]
len(store_396666)
store_396666.Shift.value_counts()
store_396666.Shift.value_counts()[:30].plot(kind='bar', title="Shifts histogram: store 396666")
plt.savefig("histogram_396666.png", bbox_inches="tight")
fig3=plt.gcf()
#resize for report
fig3.set_size_inches(15, 11)
fig3.savefig("histogram_396666_1.png")


############ Store 149087 - St Peters Street Derby: 1233 samples
store_149087 = shifts_data[shifts_data["StoreID"]==149087]
len(store_149087)
store_149087.Shift.value_counts()
store_149087.Shift.value_counts()[:30].plot(kind='bar', title="Shifts histogram: store 149087")
fig3=plt.gcf()
plt.savefig("histogram_149087.png", bbox_inches="tight")
#resize for report
fig3.set_size_inches(15, 11)
#fig3.savefig("histogram_149087_1.png")
#fig3.savefig("ff.png")


########### Store 149092 - Allenton: 1170 samples
store_149092 = shifts_data[shifts_data["StoreID"]==149092]
len(store_149092)
store_149092.Shift.value_counts()
store_149092.Shift.value_counts()[:30].plot(kind='bar', title="Shifts histogram: store 149092")
plt.savefig("histogram_149092.png", bbox_inches="tight")
fig3=plt.gcf()
#resize for report
fig3.set_size_inches(15, 11)
fig3.savefig("histogram_149092_1.png")

########### Store 149091 - Intu Centre Derby: 2822 samples
store_149091 = shifts_data[shifts_data["StoreID"]==149091]
len(store_149091)
store_149091.Shift.value_counts()
store_149091.Shift.value_counts()[:30].plot(kind='bar', title="Shifts histogram: store 149091")
plt.savefig("histogram_149091.png", bbox_inches="tight")
fig3=plt.gcf()
#resize for report
fig3.set_size_inches(15, 11)
fig3.savefig("histogram_149091_1.png")

#general frequency of shifts:
shifts_data.Shift.value_counts()[:30].plot(kind='bar', title="General shifts histogram")
plt.savefig("histogram_shifts.png", bbox_inches="tight")
fig3=plt.gcf()
#resize for report
fig3.set_size_inches(15, 11)
fig3.savefig("histogram_shifts_1.png")

#----------------------------------------------------------------------
#---------------------- Workers vs sales correlations ----------------
#----------------------------------------------------------------------

#creates a new dataset, that has for each store, each day, each hour
#the number of workers in that hour, to analyse it agains the demand, that
#is in the same format. (plot them, calculate correlation, compare the
#correlation for each store, correlation per weekday maybe be important too)

#Correlation number of workers 
sales_workers_data['Month']=sales_workers_data.NewDate.apply(lambda x: get_month(x))
sales_workers_data['Weekday']=sales_workers_data.NewDate.apply(lambda x: get_weekday(x))
sales_workers_data_nozeros = sales_workers_data[(sales_workers_data["NumberofWorkers"]!=0)]
correlations_matrices=sales_workers_data_nozeros.groupby('StoreID')[['sales','NumberofWorkers']].corr()
corr_index = correlations_matrices.index.values
corr_index_series = Series(corr_index)
storeID_corr = corr_index_series.apply(lambda x: x[0])
storeID_corr=storeID_corr.unique()
#mask_corr = (coef!=1) & (~np.isnan(coef))
coef= correlations_matrices['sales'].values
coef=coef[coef!=1]
data = {"StoreID":storeID_corr,"corr":coef}
ws_corr = DataFrame(data, columns=['StoreID','corr'])
#ws_corr.to_csv('workers_sales_corr_bystore.csv',index=False)

sales_workers_data_nozeros.groupby('Weekday')[['sales','NumberofWorkers']].corr()

#Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))

###################               SCATTERPLOTS            ################
################### Store 149091 - Intu Centre Derby: 2822 samples 0.42 total corr

scatterplots(149091,sales_workers_data)
#correlation matrix of sales vs number of workers of store 149091 grouped by weekday
sw_149091 = sales_workers_data[(sales_workers_data["StoreID"]==149091) & (sales_workers_data["NumberofWorkers"]!=0)]
sw_149091.groupby('Weekday')[['sales','NumberofWorkers']].corr()


#################  Store 149087 - St Peters Street Derby: 1233 samples 0.48 total corr
scatterplots(149087,sales_workers_data)
#correlation matrix of sales vs number of workers of store 149091 grouped by weekday
sw_149087 = sales_workers_data[(sales_workers_data["StoreID"]==149087) & (sales_workers_data["NumberofWorkers"]!=0)]
sw_149087.groupby('Weekday')[['sales','NumberofWorkers']].corr()


############ Store 396666 - Derby, Kingsway: 1619 samples
scatterplots(396666,sales_workers_data)
#correlation matrix of sales vs number of workers of store 149091 grouped by weekday
sw_396666 = sales_workers_data[(sales_workers_data["StoreID"]==396666) & (sales_workers_data["NumberofWorkers"]!=0)]
sw_396666.groupby('Weekday')[['sales','NumberofWorkers']].corr()


########### Store 149092 - Allenton: 1170 samples 0.23 total corr
scatterplots(149092,sales_workers_data)
#correlation matrix of sales vs number of workers of store 149091 grouped by weekday
sw_149092 = sales_workers_data[(sales_workers_data["StoreID"]==149092) & (sales_workers_data["NumberofWorkers"]!=0)]
sw_149092.groupby('Weekday')[['sales','NumberofWorkers']].corr()


########### Store 146633 - Allenton: 1170 samples 0.23 total corr
scatterplots(146633,sales_workers_data)
#correlation matrix of sales vs number of workers of store 149091 grouped by weekday
sw_146633 = sales_workers_data[(sales_workers_data["StoreID"]==146633) & (sales_workers_data["NumberofWorkers"]!=0)]
sw_146633.groupby('Weekday')[['sales','NumberofWorkers']].corr()


#For the second part of Ning questions:
#General max and min number of workers
sales_workers_data_g_max = sales_workers_data.groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "max"})#This is the max shifts overlap
sales_workers_data_g_min = sales_workers_data_nozeros.groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "min"})#This is the min shifts overlap
max(sales_workers_data_g_max.NumberofWorkers)#15
min(sales_workers_data_g_min.NumberofWorkers)#1
# StoreID 149087 max and min number of workers
workers_max_149087 = sales_workers_data[sales_workers_data.StoreID == 149087].groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "max"})#This is the max shifts overlap
workers_min_149087 = sales_workers_data_nozeros[sales_workers_data_nozeros.StoreID == 149087].groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "min"})#This is the min shifts overlap
max(workers_max_149087.NumberofWorkers) #4
min(workers_min_149087.NumberofWorkers) #1

# StoreID 149091 max and min number of workers
workers_max_149091 = sales_workers_data[sales_workers_data.StoreID == 149091].groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "max"})#This is the max shifts overlap
workers_min_149091 = sales_workers_data_nozeros[sales_workers_data_nozeros.StoreID == 149091].groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "min"})#This is the min shifts overlap
max(workers_max_149091.NumberofWorkers)#15
min(workers_min_149091.NumberofWorkers)#1

# StoreID 149092 max and min number of workers
workers_max_149092 = sales_workers_data[sales_workers_data.StoreID == 149092].groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "max"})#This is the max shifts overlap
workers_min_149092 = sales_workers_data_nozeros[sales_workers_data_nozeros.StoreID == 149092].groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "min"})#This is the min shifts overlap
max(workers_max_149092.NumberofWorkers)#4
min(workers_min_149092.NumberofWorkers)#1

# StoreID 396666 max and min number of workers
workers_max_396666 = sales_workers_data[sales_workers_data.StoreID == 396666].groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "max"})#This is the max shifts overlap
workers_min_396666 = sales_workers_data_nozeros[sales_workers_data_nozeros.StoreID == 396666].groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "min"})#This is the min shifts overlap
max(workers_max_396666.NumberofWorkers)#5
min(workers_min_396666.NumberofWorkers)#1

# StoreID 149087 max and min number of workers
workers_max_146633 = sales_workers_data[sales_workers_data.StoreID == 146633].groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "max"})#This is the max shifts overlap
workers_min_146633 = sales_workers_data_nozeros[sales_workers_data_nozeros.StoreID == 146633].groupby(['StoreID','NewDate']).agg({"NumberofWorkers": "min"})#This is the min shifts overlap
max(workers_max_146633.NumberofWorkers)#10
min(workers_min_146633.NumberofWorkers)#1


##########----------- DATA PREPROCESSING: DEMAND ----------#########
#--------------------- duplicates --------------------
#check for general duplicates
forecast_data[forecast_data.duplicated()]
#check for duplicates employees and days
forecast_data[forecast_data.duplicated(['StoreID', 'ScheduleStartDate','ScheduleEndingDate','DayOfWeekDate'])]
#They are only repeated values in two stores, and they are not the ones we are studying for now.
#They differ on the estimate sales... so I don't know which one is correct.
#I leave them for now.

#In this dataset I do not have the hour, I am gonna use "sales_data" but
#I think thats the actual sales
sales_data[sales_data["StoreID"]==149092]
len(sales_data)

########## ----------------- PLOTS AND CHECKS ----------------##########
#prueba.apply(lambda x: x.day).unique()
#------  prints 5 first rows
#employee_constraints.head()
#-------   check the column types
#employee_constraints.dtypes
#-------   unique values

#sales_data.shape #returns the number of rows and columns of the dataframe
#sales_data1.plot(x='Date', y='sales')
#sales_data2.plot(x='Date', y='sales')
#sales_data.plot(x='Date', y='sales')





