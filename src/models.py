from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import numpy as np


def evaluate_test_set(cross_val, df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=0)
    cv_model = cross_val.fit(train_df)
    predictions = cv_model.transform(test_df)
    regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="r2")
    print("R Squared (R2) on test data = %g" % regression_evaluator.evaluate(predictions))
    regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="rmse")
    print("RMSE on test data = %g" % regression_evaluator.evaluate(predictions))
    y_pred = predictions.select(['prediction'])
    return y_pred


def random_forest_model(df, features_col='features_scaled', label_col='ArrDelay'):
    rf = RandomForestRegressor(labelCol=label_col, featuresCol=features_col)
    param_grid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start=5, stop=11, num=2)]) \
        .build()

    cross_val = CrossValidator(estimator=rf,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(labelCol='ArrDelay', metricName='rmse'),
                               numFolds=3)
    print('Random Forest results:')
    y_pred = evaluate_test_set(cross_val, df)

    return y_pred


def decision_tree_model(df, features_col='features_scaled', label_col='ArrDelay'):
    dt = DecisionTreeRegressor(labelCol=label_col, featuresCol=features_col, maxBins=5027)
    param_grid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [5, 10]) \
        .build()
    cross_val = CrossValidator(estimator=dt,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(labelCol='ArrDelay', metricName='r2'),
                               numFolds=3)

    print('Decision Tree results:')
    y_pred = evaluate_test_set(cross_val, df)
    return y_pred


def linear_regression_model(df, features_col='features_scaled', label_col='ArrDelay'):
    lr = LinearRegression(featuresCol=features_col, labelCol=label_col, maxIter=100,  fitIntercept=True,
                          standardization=True)
    param_grid = ParamGridBuilder()\
        .addGrid(lr.regParam, [0.1, 0.01])\
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
        .build()

    cross_val = CrossValidator(estimator=lr,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(labelCol='ArrDelay', metricName='rmse'),
                               numFolds=3)
    print('Logistic Regression results:')
    y_pred = evaluate_test_set(cross_val, df)

    return y_pred
