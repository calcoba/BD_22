from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA


def pca(df, k_number=4):

    assembler = VectorAssembler(inputCols=df.drop('ArrDelay').columns, outputCol="features") #imprescindible, crea una nueva columna 'features' que es un
                                                                            # vector de las demas caracteristicas
    output = assembler.transform(df)

    output.show()

    pca_model = PCA(k=k_number, inputCol="features", outputCol="pca_features")

    #  pca.setOutputCol("pca_features")

    model = pca_model.fit(output)

    eigenvalues = model.explainedVariance
    eigenvectors = model.pc
    pca_data = model.transform(output)
    pca_data.show(5)
    assembler_pca = VectorAssembler(inputCols=pca_data.select('pca_features', 'ArrDelay').columns,
                                    outputCol="features_pca")
    pca_data = assembler_pca.transform(pca_data)
    pca_data.show(5, False)

    return eigenvalues, eigenvectors, pca_data
