#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import ml_environment as env
from dataprocess_DF import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator 

if __name__ == "__main__":
    print(sys.argv)
    sc= env.CreateSparkContext("LinearRegression")
    sqlContext = SQLContext(sc)
    env.Setargv()

    print("===========LoadData====================")
    df = Load_data(sqlContext)
    print(df.count())
    lable_name  = [df.schema.names[i]for i in env.idx_label]
    print("predicted_label=",lable_name)
    
    print("===========DataProcess====================")
    df,cat_dist = dataprocess(df)

    for idx in range(0,len(env.idx_cat)):
        for i in range(0,len(cat_dist[idx].labels)):
            print("idx_cat"+ str(env.idx_cat[idx])+" "+str(i)+':'+cat_dist[idx].labels[i])
    
    print("===========SplitData====================")
    train_df, test_df = df.randomSplit(env.split_prop)
    
    print("===========VectorAssembler====================")
    feature = df.columns[1:len(df.columns)-1]
    assembler = VectorAssembler(inputCols=feature, outputCol="features")
    
    print("=============pipeline==================")
    model = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,labelCol=lable_name[0], featuresCol="features")
    pipeline = Pipeline(stages=[assembler,model])
    pipeline.getStages()
    
    print("===========TaintingAndTesting====================")
    pipelineModel = pipeline.fit(train_df)
    predicted=pipelineModel.transform(test_df)
    
    
    print("===========PredictedScore====================")
    evaluator = RegressionEvaluator(labelCol=lable_name[0])
    
    R_squared = evaluator.evaluate(predicted, {evaluator.metricName: "r2"})
    RMSE = evaluator.evaluate(predicted, {evaluator.metricName: "rmse"})
    
    print("RMSE",RMSE,"R_squared",R_squared)