from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor, LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler


def split_data(df_to_split):
    train_df_data, test_df_data = df_to_split.randomSplit([0.8, 0.2], seed=0)
    return train_df_data, test_df_data


def linea_regression_model(df, vectorization=True, features_col=None):
    if vectorization:
        features_col = "features"
        assembler = VectorAssembler(inputCols=df.drop('ArrDelay').columns, outputCol=features_col) #imprescindible, crea una nueva columna 'features' que es un
                                                                            # vector de las demas caracteristicas
        df = assembler.transform(df).select('ArrDelay', 'features')
    df.show(5, False)
    lr = LinearRegression(featuresCol=features_col, labelCol='ArrDelay', maxIter=100,  fitIntercept=True,
                          standardization=True)
    train_df, test_dt = split_data(df)
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

    cross_val = CrossValidator(estimator=pipeline,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(labelCol='ArrDelay', metricName='rmse'),
                               numFolds=3)
    train_df, test_dt = split_data(df)
    cv_model = cross_val.fit(train_df)
    predictions = cv_model.transform(test_dt)
    regression_evaluator = RegressionEvaluator(labelCol='ArrDelay', metricName='rmse')
    print('Random Forest classifier Accuracy:', regression_evaluator.evaluate(predictions))
    y_true = predictions.select(['ArrDelay'])
    y_pred = predictions.select(['prediction'])
    print('Done')

    return y_true, y_pred


def DecisionTree(df):
    assembler = VectorAssembler(inputCols=df.drop('ArrDelay').columns, outputCol="features")
    dt = DecisionTreeRegressor(labelCol='ArrDelay', featuresCol='features', maxBins=5027)
    train_df, test_df = split_data(df)
    pipeline = Pipeline(stages=[assembler, dt])
    param_grid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [5, 10]) \
        .addGrid(dt.minInstancesPerNode, [5, 10]) \
        .build()
    cross_val = CrossValidator(estimator=pipeline,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(labelCol='ArrDelay', metricName='r2'),
                               numFolds=4)
    train_df, test_df = split_data(df)
    cv_model = cross_val.fit(train_df)
    predictions = cv_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="r2")
    print("R Squared (R2) on test data = %g" % regression_evaluator.evaluate(predictions))
    regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="rmse")
    print("RMSE on test data = %g" % regression_evaluator.evaluate(predictions))
