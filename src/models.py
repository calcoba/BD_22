from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
import numpy as np
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.regression import LinearRegression
def LinearRegression(df):

    assembler = VectorAssembler(inputCols=df.columns, outputCol="features") #imprescindible, crea una nueva columna 'features' que es un
                                                                            # vector de las demas caracteristicas                                                                            
    lr=LinearRegression(featuresCol='features', labelCol='ArrDelay', predictionCol='y_pred', 
                        maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-06, fitIntercept=True, standardization=True, 
                        solver='auto', weightCol=None, aggregationDepth=2, loss='squaredError', epsilon=1.35, maxBlockSizeInMB=0.0)

    model = lr.fit(df)
    coeficients = model.coeficients

    return coeficients




def RandomForest(df):
    assembler = VectorAssembler(inputCols=df.drop('ArrDelay').columns, outputCol="features")
    vectorized_df = assembler.transform(df).select("ArrDelay", "features")
    rf = RandomForestRegressor(labelCol="ArrDelay", featuresCol="features")
    pipeline = Pipeline(stages=[assembler, rf])
    param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [int(x) for x in np.linspace(start=70, stop=100, num=10)]) \
        .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start=5, stop=11, num=2)]) \
        .build()

    cross_val = CrossValidator(estimator=rf,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(labelCol='ArrDelay', metricName='rmse'),
                               numFolds=3)
    (trainingData, testData) = vectorized_df.randomSplit([0.8, 0.2])
    cv_model = cross_val.fit(trainingData)
    predictions = cv_model.transform(testData)
    regression_evaluator = RegressionEvaluator(labelCol='ArrDelay', metricName='rmse')
    print('Random Forest classifier Accuracy:', regression_evaluator.evaluate(predictions))
    y_true = predictions.select(['ArrDelay']).collect()
    y_pred = predictions.select(['prediction']).collect()

    return y_true, y_pred
