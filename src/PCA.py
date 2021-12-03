from pyspark.ml.feature import PCA


def pca(df, k_number=10):
    pca_model = PCA(k=k_number, inputCol="features_scaled", outputCol="pca_features")
    model = pca_model.fit(df)
    pca_data = model.transform(df).select('pca_features', 'ArrDelay')
    pca_data.show(5)

    eigenvalues = model.explainedVariance
    eigenvectors = model.pc

    return eigenvalues, eigenvectors, pca_data
