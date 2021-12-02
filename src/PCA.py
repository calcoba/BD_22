from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA

def pca(df, k=4):

    assembler = VectorAssembler(inputCols=df.columns, outputCol="features") #imprescindible, crea una nueva columna 'features' que es un
                                                                            # vector de las demas caracteristicas
    output = assembler.transform(df)

    output.show()

    pca = PCA(k, inputCol="features")

    pca.setOutputCol("pca_features")

    model = pca.fit(output)

    eigenvalues = model.explainedVariance
    eigenvectors = model.pc

    return eigenvalues, eigenvectors