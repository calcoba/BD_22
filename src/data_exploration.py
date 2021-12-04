import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import time
import seaborn as sns


def plot_corr_matrix(correlations, attr, name):
    """This function will display a plot of the correlation between the different variables presented in the
    dataframe.
        :parameter
            correlations: correlation matrix with the values for the correlation between each pairs of variables.
            attr: list of the different features names used for computing the correlation matrix.
    """
    plt.figure(figsize=(20, 15))
    plt.tight_layout()
    plt.title("Correlation Matrix for Specified Attributes")
    sns.heatmap(correlations, annot=True, cbar=True, xticklabels=attr, yticklabels=attr)
    plt.yticks(rotation=0)
    plt.savefig('data/'+str(name)+'.png',  dpi='figure', format='png')
    plt.close()


def compute_corr(data_base, attr, display=True, name=None):
    """This function will compute the correlation matrix between the different features present in the dataset. The
    function takes the complete set of features and perform a vectorization operation with the VectorAssembler function
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
    if not name:
        name = np.random.randint(0, 10)

    if display:
        plot_corr_matrix(corr_matrix, attr, name)

    return corr_matrix
