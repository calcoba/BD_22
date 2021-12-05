from pyspark.ml.regression import LinearRegressionModel, DecisionTreeRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator


def generate_predictions(model_path, model_name, df):
    if model_name == 'lr':
        model = LinearRegressionModel.load(model_path)
        features = 'features_scaled'
        coefficients_lr = model.coefficients
        print(coefficients_lr)
    elif model_name == 'dt':
        model = DecisionTreeRegressionModel.load(model_path)
        features = 'features_scaled'
    elif model_name == 'lr_pca':
        model = LinearRegressionModel.load(model_path)
        features = 'pca_features'
        coefficients_lr = model.coefficients
        print(coefficients_lr)
    predictions = model.transform(df.select(features, 'ArrDelay'))
    predictions.show(5, True)
    regression_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="r2")
    regression_evaluator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="rmse")
    results = ["  Model results:",
               "    R Squared (R2) on test data = %g" % regression_evaluator_r2.evaluate(predictions),
               "    RMSE on test data = %g" % regression_evaluator_rmse.evaluate(predictions)]

    for line in results:
        print(line)
    return predictions
