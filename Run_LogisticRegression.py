#!/usr/bin/python
# -*- coding:utf-8 -*-
import ml_environment as env
from dataprocess_DF import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
 

if __name__ == "__main__":
    sc= env.CreateSparkContext("LogisticRegressionClassifier")
    sqlContext = SQLContext(sc)
    env.Setargv()

    print("===========LoadData====================")
    df = Load_data(sqlContext)
    print(df.count())
    
    print("===========DataProcess====================")
    
    if env.recode_label == True:
        df,cat_dist = dataprocess(df,recode=env.idx_label)
    else:
        df,cat_dist = dataprocess(df)
    
    for idx in range(0,len(env.idx_cat)):
        for i in range(0,len(cat_dist[idx].labels)):
            print("idx_cat"+ str(env.idx_cat[idx])+" "+str(i)+':'+cat_dist[idx].labels[i]) 
    
    print("===========SplitData====================")
    train_df, test_df = df.randomSplit(env.split_prop)
    
    print("===========VectorAssembler====================")
    feature = df.columns[1:len(df.columns)-1]
    lable_name = df.columns[-1]
    print(lable_name)
    assembler = VectorAssembler(inputCols=feature, outputCol="features")
    
    print("=============pipeline==================")
    model = LogisticRegression(regParam=0.1, labelCol=lable_name, featuresCol="features" , family ='binomial')
    pipeline = Pipeline(stages=[assembler,model])
    pipeline.getStages()
    
    print("===========TaintingAndTesting====================")
    pipelineModel = pipeline.fit(train_df)
    predicted=pipelineModel.transform(test_df)
    
    print("===========PredictedAUC====================")
    evaluator = BinaryClassificationEvaluator(
                              rawPredictionCol="rawPrediction",
                              labelCol= lable_name,  
                              metricName="areaUnderROC"  )
    auc= evaluator.evaluate(predicted)
    print(auc)
    
    print("===========PredictedScore====================")
    Multi_evaluator = MulticlassClassificationEvaluator(labelCol= lable_name)
    Accuracy= Multi_evaluator.evaluate(predicted ,{evaluator.metricName: "accuracy"})
    Precision = Multi_evaluator.evaluate(predicted ,{evaluator.metricName: "weightedPrecision"})
    Recall = Multi_evaluator.evaluate(predicted ,{evaluator.metricName: "weightedRecall"})
    F1 = Multi_evaluator.evaluate(predicted ,{evaluator.metricName: "f1"})

    print("Accuracy",Accuracy,"Precision",Precision,"Recall",Recall,"F1",F1)