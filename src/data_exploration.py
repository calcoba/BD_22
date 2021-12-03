import matplotlib.pyplot as plt
import numpy as np
import pyspark.sql
from pyspark.ml.stat import Correlation, ChiSquareTest, KolmogorovSmirnovTest
from pyspark.ml.feature import VectorAssembler


def plot_corr_matrix(correlations, attr, fig_no):
    fig = plt.figure(fig_no, figsize=(30,30))
    ax = fig.add_subplot(111)
    ax.set_title("Correlation Matrix for Specified Attributes")
    cax = ax.matshow(correlations, vmax=1, vmin=-1)
    for (i, j), z in np.ndenumerate(correlations):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize=30,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(attr)), labels=attr, fontsize=30, rotation=90)
    ax.set_yticks(np.arange(len(attr)), labels=attr, fontsize=30)
    plt.show()


def compute_corr(data_base, attr, vectorize=True):
    # convert to vector column first

    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=data_base.columns, outputCol=vector_col)
    df_vector = assembler.transform(data_base).select(vector_col)
    df_vector.show(5, False)
    print(vector_col)
    # get correlation matrix
    pearson_corr = Correlation.corr(df_vector, vector_col, 'pearson').collect()[0][0]
    # print(pearson_corr)
    corr_matrix = pearson_corr.toArray().tolist()

    plot_corr_matrix(corr_matrix, attr, 234)


def compute_ChiSquared(spark, data_base, vectorize=True):
    if vectorize:
        vector_col = "features"
        assembler = VectorAssembler(inputCols=data_base.drop('ArrDelay').columns, outputCol=vector_col)
        vectorized_df = assembler.transform(data_base).select("ArrDelay", "features")
    else:
        vectorized_df = data_base
        vector_col = "pca_features"
    vectorized_df.show(5)
    chiSquares_values = ChiSquareTest.test(vectorized_df, vector_col, "ArrDelay")
    chiSquares_values.show()
    return chiSquares_values
