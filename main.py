import src.load_data as load_data
from src import data_exploration
import decompress
from pyspark.sql import SparkSession
from src import models, PCA
import glob


if __name__ == '__main__':
    # spark = SparkSession.builder.appName('big_data_project').getOrCreate()
    spark = SparkSession.builder \
        .appName('big_data_project') \
        .config("spark.driver.memory", "14g") \
        .getOrCreate()

    path = './data/'
    '''compressed_file_path = glob.glob(path+'*.csv.bz2')
    decompress.decompress(compressed_file_path)'''

    #files = glob.glob(path+'*.csv')
    #for file_name in files:
    plane_db = load_data.load_data(spark, path+'*.csv')

    data_exploration.compute_corr(plane_db.drop('features', 'features_scaled'),
                                  plane_db.drop('features', 'features_scaled').columns)
    '''chiSquared_results = data_exploration.compute_ChiSquared(spark, plane_db)
    print(str(chiSquared_results.select('statistics').show(truncate=False)))'''

    eigenvalues, eigenvectors, pca_data = PCA.pca(plane_db.select('features_scaled', 'ArrDelay'))
    '''print(eigenvalues, eigenvectors)
    pca_data.show(5)'''
    data_exploration.compute_corr(pca_data.select('pca_features', 'ArrDelay'),
                                  ['PCA_0', 'PCA_1', 'PCA_2', 'PCA_3', 'PCA_4',
                                   'PCA_5', 'PCA_6', 'PCA_7', 'PCA_8', 'PCA_9', 'ArrDelay'], False)

    '''y_true, y_pred = models.RandomForest(plane_db.drop('UniqueCarrier_index', 'TailNum_index',
                                                       'Origin_index', 'Dest_index'))'''
    lr_coefficients = models.linea_regression_model(plane_db.select('features_scaled', 'ArrDelay'))
    '''decision_tree_classifier = models.decision_tree_model(plane_db.select('features_scaled', 'ArrDelay'))'''
    lr_coefficients_pca = models.linea_regression_model(pca_data, features_col="pca_features")
