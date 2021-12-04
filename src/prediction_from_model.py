from pyspark.ml.regression import LinearRegressionModel, GBTRegressionModel, DecisionTreeRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator


def generate_predictions(model_path, model_name, df):
    if model_name == 'lr':
        model = LinearRegressionModel.load(model_path)
        features = 'features_scaled'
    elif model_name == 'gbt':
        model = GBTRegressionModel.load(model_path)
        features = 'features_scaled'
    elif model_name == 'dt':
        model = DecisionTreeRegressionModel.load(model_path)
        features = 'features_scaled'
    elif model_name == 'lr_pca':
        model = LinearRegressionModel.load(model_path)
        features = 'pca_features'
    predictions = model.transform(df.select(features, 'ArrDelay'))
    regression_evaluator_r2 = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="r2")
    regression_evaluator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol="ArrDelay", metricName="rmse")
    results = ["  Model results:",
               "    R Squared (R2) on test data = %g" % regression_evaluator_r2.evaluate(predictions),
               "    RMSE on test data = %g" % regression_evaluator_rmse.evaluate(predictions)]

    for line in results:
        print(line)
    return predictions
