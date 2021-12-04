import src.load_data as load_data
from src import data_exploration
import decompress
from pyspark.sql import SparkSession
from src import models, PCA
import glob
import sys

if __name__ == '__main__':
    # spark = SparkSession.builder.appName('big_data_project').getOrCreate()
    spark = SparkSession.builder \
        .appName('big_data_project') \
        .config("spark.driver.memory", "14g") \
        .getOrCreate()

    ### Locating, decompressing and loading data ###
    if len(sys.argv) > 1:
        path = sys.argv[1]+'/'
        print(path)
    compressed_folder_path = glob.glob(path + '*.zip')
    if not compressed_folder_path:  # This means that there aren't zip folders in the data directory
        compressed_file_path = glob.glob(path + '*.csv.bz2')
        print('No zip folders are located in the "data" folder.')
        if not compressed_file_path:  # This means that there aren't bz2 files in the data directory
            print('No bz2 files are located in the "data" folder.')
        else:
            decompress.decompress_bz2(compressed_file_path)  # Decompress bz2 files
    else:
        decompress.decompress_zip(compressed_folder_path)  # Extract bz2 files from the zip folder
        compressed_file_path = glob.glob(path + '*.csv.bz2')
        decompress.decompress_bz2(compressed_file_path)  # Decompress bz2 files

    plane_db = load_data.load_data(spark, path + '*.csv')

    correlation_matrix = data_exploration.compute_corr(plane_db.drop('features', 'features_scaled'),
                                                       plane_db.drop('features', 'features_scaled').columns)

    eigenvalues, eigenvectors, pca_data = PCA.pca(plane_db.select('features_scaled', 'ArrDelay'))
    
    correlation_matrix_pca = data_exploration.compute_corr(pca_data.select('pca_features', 'ArrDelay'),
                                                           ['PCA_0', 'PCA_1', 'PCA_2', 'PCA_3', 'PCA_4',
                                                            'PCA_5', 'PCA_6', 'PCA_7', 'PCA_8', 'PCA_9', 'ArrDelay'])

    y_pred_gbt, gbt_data = models.GBT_regressor_model(plane_db.select('features_scaled', 'ArrDelay'))
    y_pred_lr, lr_data = models.linear_regression_model(plane_db.select('features_scaled', 'ArrDelay'))
    for line in lr_data:
        print(line)
    y_pred_dt, dt_data = models.decision_tree_model(plane_db.select('features_scaled', 'ArrDelay'))
    y_pred_lr_pca, lr_pca_data = models.linear_regression_model(pca_data, features_col="pca_features")

    complete_results = []
    complete_results.extend(gbt_data)
    complete_results.extend(lr_data)
    complete_results.extend(dt_data)
    complete_results.extend(lr_pca_data)
    with open('results.txt', 'w') as file_name:
        for line in complete_results:
            file_name.write(line+'\n')
        file_name.close()
