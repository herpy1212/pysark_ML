#!/usr/bin/python
# -*- coding:utf-8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import  StringIndexer, OneHotEncoder,VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf,col
from pyspark.sql import SQLContext
import ml_environment as env
from pyspark.sql.functions import lit


def Load_data(sqlContext):
    df = sqlContext.read.format("csv") \
    .option("header", env.Header) \
    .option("delimiter", env.delimiter) \
    .load(env.Path+env.train_fname) 
    
    return df


def category_to_vactor(idx,df):
    cat_dist = []
    for i in idx:
        cat = df.schema.names[i]
        #StringToIndex
        categoryIndexer = StringIndexer(inputCol=cat,outputCol="Index_"+cat)
        categoryTransformer = categoryIndexer.fit(df)
        new_df=categoryTransformer.transform(df)
        #OneHotEncoder
        encoder = OneHotEncoder(dropLast=False,inputCol="Index_"+cat,outputCol="Vector_"+cat)
        new_df=encoder.transform(new_df)
        df = new_df
#         cat_dist = [cat_dist,categoryTransformer]
        cat_dist.append(categoryTransformer)
    return (new_df,cat_dist)


def replace_question(x,symbol):
    return ("0" if x==symbol else x)
replace_question= udf(replace_question)

def convert_negative(x):
    return (0 if float(x) < 0 else x)
convert_negative= udf(convert_negative)


def select_filter_data(df,idx_index,idx_cat,idx_num,idx_label):
    filter_df= df.select(\
                [df.schema.names[i] for i in idx_index] + \
                [df.schema.names[i] for i in idx_cat] + \
                [replace_question(col(column),lit(env.symbol)).cast("double").alias(column)  
                for column in [df.columns[i] for i in idx_num ]]+\
                [replace_question(col(column),lit(env.symbol)).cast("double").alias(column)  
                for column in [df.columns[i] for i in idx_label]])
    return filter_df


def select_filter_data_positive(df,idx_index,idx_cat,idx_num,idx_label):
    filter_df= df.select(\
                [df.schema.names[i] for i in idx_index] + \
                [df.schema.names[i] for i in idx_cat] + \
                [convert_negative(replace_question(col(column),lit(env.symbol))).cast("double").alias(column)  
                for column in [df.columns[i] for i in idx_num ]]+\
                [replace_question(col(column),lit(env.symbol)).cast("double").alias(column)  
                for column in [df.columns[i] for i in idx_label]])
    return filter_df

def drop_Null(df):
    name_list = df.schema.names
    idx = range(0,len(name_list))
    new_df = df.na.drop(subset=[name_list[i] for i in idx])
    return new_df   



def dataprocess(df,positive=False,recode = False):
    df_size = len(df.columns)
    (df,cat_dist) = category_to_vactor(env.idx_cat,df)
    new_idx_cat = [(df_size-1)+(i+1)*2 for i in range(0,len(env.idx_cat))]
    
    idx_label = env.idx_label
    
    if recode != False:
        (df,recode_dist) = category_to_vactor(recode,df)
        idx_label = [(df_size-1)+(len(env.idx_cat)+1)*2-1]
        for i in range(0,len(recode_dist[0].labels)):
            print("recode_label"+" "+str(i)+':'+recode_dist[0].labels[i])
         
    if positive == True:
        df = select_filter_data_positive(df,env.idx_index,new_idx_cat,env.idx_num,idx_label)
    else:
        df = select_filter_data(df,env.idx_index,new_idx_cat,env.idx_num,idx_label)
    
    if env.Null_drop == True:
        df = drop_Null(df)
    
    return df,cat_dist