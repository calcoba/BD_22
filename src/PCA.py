from pyspark.ml.feature import PCA, PCAModel


def pca(df, k_number=5, validation=False):
    if not validation:
        pca_model = PCA(k=k_number, inputCol="features_scaled", outputCol="pca_features")
        model = pca_model.fit(df)
        model.write().overwrite().save('results/pca_model/')
        pca_data = model.transform(df).select('pca_features', 'ArrDelay')
    else:
        model = PCAModel.load('results/pca_model/')
        pca_data = model.transform(df).select('pca_features', 'ArrDelay')
    pca_data.show(5)

    eigenvalues = model.explainedVariance
    eigenvectors = model.pc

    return eigenvalues, eigenvectors, pca_data
