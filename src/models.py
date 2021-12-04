from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor, LinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import numpy as np


def evaluate_test_set(cross_val, df):
    """This function is used for implementing the different cross-validation models available.
    First the function will divide the database presented in two subset, a train set, used for fitting the model, and
    a test set, used for measuring the model performance. As it is a cross-validation model fitting will be done for
    each set of parameters passed in the cross_val variable, with also cross validation for selecting the best
    parameter set.
    The function will plot two measures selected for assessing the performance of the model, the coefficient
    of linear determination (r2) and the Root Mean Squared Error (rmse). For evaluating this models
    the RegressionEvaluator function from pyspark is used.
        :parameter
            cross_val: the Cross Validator model to be fitted and transformed.
            df: dataframe type variable. The dataframe, vectorized, for implementing the model. A column with the
            features and another with the target variable (ArrDelay).
        :returns
            y_pred: the model prediction for the test set.
            best_model: the bes model encountered after the cross-validation is performed.
    """

    train_df, test_df = df.randomSplit([0.9, 0.1], seed=0)
    cv_model = cross_val.fit(train_df)
    predictions = cv_model.transform(test_df)
    regression_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="r2")
    regression_evaluator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="rmse")

    print("R Squared (R2) on test data = %g" % regression_evaluator_r2.evaluate(predictions))
    print("RMSE on test data = %g" % regression_evaluator_rmse.evaluate(predictions))

    y_pred = predictions.select(['prediction'])
    best_model = cv_model.bestModel

    return y_pred, best_model


def decision_tree_model(df, features_col='features_scaled', label_col='ArrDelay'):
    """This function will implement a cross-validated decision tree model with parameter grid search, to be passed
    to the evaluation function.
    The model created will perform a grid search for the maximum depth and maximum bins parameters with a
    cross-validation of 3.
    Once the evaluation is performed the function will plot the best parameter found during the fit.
        :parameter
            df: dataframe variable with the database to be used in the model. It is in vectorized form with a features
            column and a label column (ArrDelay)
            features_col: optional with default value 'features_scaled', is the name of the column in the dataframe
            where the independent variables are stored in vectorized form.
            label_col: optional with default value 'ArrDelay', the column name of the target variable.
        :returns
            y_pred: the prediction made for the test set.
    """

    dt = DecisionTreeRegressor(labelCol=label_col, featuresCol=features_col)
    param_grid = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [5, 10, 15]) \
        .addGrid(dt.maxBins, [20, 40, 80])\
        .build()
    cross_val = CrossValidator(estimator=dt,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(labelCol='ArrDelay', metricName='r2'),
                               numFolds=3)

    print('Decision Tree results:')

    y_pred, model = evaluate_test_set(cross_val, df)

    print('Best model has parameters:')
    print('Maximum depth parameter: ', model.getMaxDepth())
    print('Maximum bins parameter: ', model.getMaxBins())

    return y_pred


def linear_regression_model(df, features_col='features_scaled', label_col='ArrDelay'):
    """This function will implement a cross-validated logistic regression model with parameter grid search,
    to be passed to the evaluation function.
    The model created will perform a grid search for the regression and elastic net parameters with a
    cross-validation of 3.
    Once the evaluation is performed the function will plot the best parameter found during the fit.
        :parameter
            df: dataframe variable with the database to be used in the model. It is in vectorized form with a features
            column and a label column (ArrDelay).
            features_col: optional with default value 'features_scaled', is the name of the column in the dataframe
            where the independent variables are stored in vectorized form.
            label_col: optional with default value 'ArrDelay', the column name of the target variable.
        :returns
            y_pred: the prediction made for the test set.
    """

    lr = LinearRegression(featuresCol=features_col, labelCol=label_col, maxIter=100,  fitIntercept=True)
    param_grid = ParamGridBuilder()\
        .addGrid(lr.regParam, [0.1, 0.01, 0.001])\
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
        .build()
    cross_val = CrossValidator(estimator=lr,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(labelCol='ArrDelay', metricName='r2'),
                               numFolds=3)
    print('Logistic Regression results:')

    y_pred, model = evaluate_test_set(cross_val, df)

    print('Best model has parameters:')
    print('Reg parameter: ', model.getRegParam())
    print('Elastic Net parameter: ', model.getElasticNetParam())

    return y_pred


def GBT_regressor_model(df, features_col='features_scaled', label_col='ArrDelay'):
    """This function will implement a cross-validated GBT regression model with parameter grid search,
    to be passed to the evaluation function.
    The model created will perform a grid search for the maximum depth and subsampling parameters with a
    cross-validation of 3.
    Once the evaluation is performed the function will plot the best parameter found during the fit.
        :parameter
            df: dataframe variable with the database to be used in the model. It is in vectorized form with a features
            column and a label column (ArrDelay)
            features_col: optional with default value 'features_scaled', is the name of the column in the dataframe
            where the independent variables are stored in vectorized form.
            label_col: optional with default value 'ArrDelay', the column name of the target variable.
        :returns
            y_pred: the prediction made for the test set
    """

    gbt = GBTRegressor(featuresCol=features_col, labelCol=label_col, seed=0)
    param_grid = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [5, 10, 15]) \
        .addGrid(gbt.subsamplingRate, [0.7, 0.8, 1]) \
        .build()
    cross_val = CrossValidator(estimator=gbt,
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(labelCol='ArrDelay', metricName='r2'),
                               numFolds=3)

    print('GTB regression results:')

    y_pred, model = evaluate_test_set(cross_val, df)

    print('Best model has parameters:')
    print('Maximum depth parameter: ', model.getMaxDepth())
    print('Subsampling Rate parameter: ', model.getSubsamplingRate())

    return y_pred
