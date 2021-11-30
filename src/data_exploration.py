import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler


def plot_corr_matrix(correlations, attr, fig_no):
    fig = plt.figure(fig_no, figsize=(30,30))
    ax = fig.add_subplot(111)
    ax.set_title("Correlation Matrix for Specified Attributes")
    ax.set_xticks(np.arange(len(attr)))
    ax.set_yticks(np.arange(len(attr)))
    ax.set_xticklabels(attr, fontsize=30, rotation=45)
    ax.set_yticklabels(attr, fontsize=30)
    for (i, j), z in np.ndenumerate(correlations):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize=30,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    cax = ax.matshow(correlations, vmax=1, vmin=-1)
    fig.colorbar(cax)
    plt.show()


def compute_corr(data_base):
    # convert to vector column first
    vector_col = "corr_features"
    plane_corr_db = data_base.drop('UniqueCarrier', 'TailNum', 'Origin', 'Dest', 'Cancelled')
    assembler = VectorAssembler(inputCols=plane_corr_db.columns, outputCol=vector_col)
    df_vector = assembler.transform(plane_corr_db).select(vector_col)

    # get correlation matrix
    pearson_corr = Correlation.corr(df_vector, vector_col, 'pearson').collect()[0][0]
    print(pearson_corr)
    corr_matrix = pearson_corr.toArray().tolist()

    plot_corr_matrix(corr_matrix, plane_corr_db.columns, 234)
