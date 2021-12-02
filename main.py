import src.load_data as load_data
from src import data_exploration
from pyspark.sql import SparkSession
from src import models, PCA

if __name__ == '__main__':
    # spark = SparkSession.builder.appName('big_data_project').getOrCreate()
    spark = SparkSession.builder \
        .appName('big_data_project') \
        .config("spark.driver.memory", "14g") \
        .getOrCreate()
    path = 'data/'
    files = ['2006']
    for file_name in files:
        plane_db, target_db = load_data.load_data(spark, path+file_name+'.csv')

    '''data_exploration.compute_corr(plane_db, plane_db.columns)
    plane_db.printSchema()'''
    '''chiSquared_results = data_exploration.compute_ChiSquared(spark, plane_db)
    print(str(chiSquared_results.select('statistics').show(truncate=False)))'''

    eigenvalues, eigenvectors, pca_data = PCA.pca(plane_db)
    print(eigenvalues, eigenvectors)
    pca_data.show(5)
    '''data_exploration.compute_corr(pca_data.select('pca_features', 'ArrDelay'),
                                  ['PCA_0', 'PCA_1', 'PCA_2', 'PCA_3', 'ArrDelay'], False)'''

    '''y_true, y_pred = models.RandomForest(plane_db.drop('UniqueCarrier_index', 'TailNum_index',
                                                       'Origin_index', 'Dest_index'))'''
    lr_coefficients = models.linea_regression_model(plane_db)
    print(lr_coefficients)
    lr_coefficients_pca = models.linea_regression_model(pca_data, False, "pca_features")
    print(lr_coefficients_pca)
