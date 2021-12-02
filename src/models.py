from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
import numpy as np
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


def linea_regression_model(df, vectorization=True, features_col=None):
    if vectorization:
        features_col = "features"
        assembler = VectorAssembler(inputCols=df.drop('ArrDelay').columns, outputCol=features_col) #imprescindible, crea una nueva columna 'features' que es un
                                                                            # vector de las demas caracteristicas
        df = assembler.transform(df).select('ArrDelay', 'features')
    df.show(5, False)
    lr = LinearRegression(featuresCol=features_col, labelCol='ArrDelay', maxIter=100,  fitIntercept=True,
                          standardization=True)
    splits = df.randomSplit([0.7, 0.3])
    train_df = splits[0]
    test_df = splits[1]
    train_df.show(5, False)
    model = lr.fit(train_df)
    # predictions = model.trans
    coefficients = model.coefficients
    lr_predictions = model.transform(test_df)
    lr_predictions.select("prediction", "ArrDelay", features_col).show(5, False)
    from pyspark.ml.evaluation import RegressionEvaluator
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="r2")
    print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="rmse")
    print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

    return coefficients




def RandomForest(df):
    assembler = VectorAssembler(inputCols=df.drop('ArrDelay').columns, outputCol="features")
    vectorized_df = assembler.transform(df).select("ArrDelay", "features")
    rf = RandomForestRegressor(labelCol="ArrDelay", featuresCol="features")
    pipeline = Pipeline(stages=[assembler, rf])
    param_grid = ParamGridBuilder() \
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
    y_true = predictions.select(['ArrDelay'])
    y_pred = predictions.select(['prediction'])
    print('Done')

    return y_true, y_pred
