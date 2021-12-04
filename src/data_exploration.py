import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.stat import Correlation, ChiSquareTest
from pyspark.ml.feature import VectorAssembler


def plot_corr_matrix(correlations, attr):
    """This function will display a plot of the correlation between the different variables presented in the
    dataframe.
        :parameter
            correlations: correlation matrix with the values for the correlation between each pairs of variables.
            attr: list of the different features names used for computing the correlation matrix.
    """

    fig = plt.figure(1, figsize=(30, 30))
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


def compute_corr(data_base, attr, display=True):
    """This function will compute the correlation matrix between the different features present in the dataset. The
    function takes the complete set of features and perform a vectorization operation witht he VectorAssembler function
    from pyspark module. After that it call the function for plotting the matrix if it is set.
        :parameter
            data_base: the dataframe with the complete set of variables. One variable for each of the columns, with
            the dependent variable also present.
            attr: a list with the names of the different features names in the same order as in the dataframe.
            display: optional boolean variable with default value True. If it is true the correlation matrix will be
            plotted.
        :return
            corr_matrix: the correlation matrix computed for the database passed.
    """

    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=data_base.columns, outputCol=vector_col)
    df_vector = assembler.transform(data_base).select(vector_col)

    # get correlation matrix
    pearson_corr = Correlation.corr(df_vector, vector_col, 'pearson').collect()[0][0]
    # print(pearson_corr)
    corr_matrix = pearson_corr.toArray().tolist()

    if display:
        plot_corr_matrix(corr_matrix, attr)

    return corr_matrix
