from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder
import numpy as np
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

def RandomForest(df):
    assembler = VectorAssembler(inputCols=df.columns, outputCol="features") #imprescindible, crea una nueva columna 'features' que es un
                                                                            # vector de las demas caracteristicas
    rf = RandomForestClassifier(labelCol="ArrDelay", featuresCol="features")
    pipeline = Pipeline(stages=[assembler, rf])
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 70, stop = 100, num = 10)]) \
        .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 11, num = 2)]) \
        .build()

    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=MulticlassClassificationEvaluator(),
                            numFolds=3)
    (trainingData, testData) = df.randomSplit([0.8, 0.2])
    cvModel = crossval.fit(trainingData)
    predictions = cvModel.transform(testData)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'label', metricName = 'accuracy')
    print('Random Forest classifier Accuracy:', multi_evaluator.evaluate(predictions))
    y_true = predictions.select(['label']).collect()
    y_pred = predictions.select(['prediction']).collect()

    return y_true,y_pred