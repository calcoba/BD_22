from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.ml import Pipeline


def pca(df, k_number=4):

    assembler = VectorAssembler(inputCols=df.drop('ArrDelay').columns, outputCol="features") #imprescindible, crea una nueva columna 'features' que es un
                                                                            # vector de las demas caracteristicas
    pca_model = PCA(k=k_number, inputCol="features", outputCol="pca_features")

    #  pca.setOutputCol("pca_features")

    pipeline = Pipeline(stages=[assembler, pca_model])
    model = pipeline.fit(df)
    pca_data = model.transform(df).select('pca_features', 'ArrDelay')
    pca_data.show(5)

    eigenvalues = model.stages[-1].explainedVariance
    eigenvectors = model.stages[-1].pc

    return eigenvalues, eigenvectors, pca_data
