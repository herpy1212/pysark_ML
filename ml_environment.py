#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
from pyspark import SparkConf, SparkContext

def Setargv():
    global train_fname,delimiter,Header,idx_index,   \
            idx_label,idx_num,idx_cat,Null_drop, \
            symbol,recode_label,split_prop
    
#     train_fname = "data/train.tsv"
    train_fname = "data/mimic_data.csv"
    delimiter=","          #default = ","
    Header="true"           #default = T  
    Null_drop = True        #default = T
    symbol = "?"            
    recode_label = False     #default = F
    split_prop = [0.9, 0.1]
    idx_index = [0]          #unique key
    idx_label = [49]             #predict label 
    idx_num = [2,6,11,12,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
    idx_cat = [1, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 50, 51, 52]
    ### classification test index
#     idx_label = [26]             #predict label 
#     idx_num  = [4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25]    #numerical feature
#     idx_cat  = [3,17]            #categorical feature
    ### regression test index
#     idx_label = [22]             #predict label 
#     idx_num  = [4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,23,24,25]    #numerical feature
#     idx_cat  = [3,17,26]

  

def SetPath(sc):
    global Path
    local_Path = "file:/home/user/pythonwork/PythonProject/"
    HDFS_Path = "hdfs://master:9000/user/"
    if sc.master[0:5] == "local" :
        Path = local_Path
    else:
        Path = HDFS_Path
        
        
        
def CreateSparkContext(AppName):
    sparkConf = SparkConf()                                                       \
                         .setAppName(AppName)        \
                         .set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    print ("master="+sc.master)
    SetPath(sc)
    return (sc)





