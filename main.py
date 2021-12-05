import src.load_data as load_data
from src import data_exploration, models, PCA, prediction_from_model
import decompress
from pyspark.sql import SparkSession
import glob
import sys

if __name__ == '__main__':
    # spark = SparkSession.builder.appName('big_data_project').getOrCreate()
    spark = SparkSession.builder \
        .appName('big_data_project') \
        .config("spark.driver.memory", "14g") \
        .getOrCreate()

    if len(sys.argv) == 3:
        validation = True
    else:
        validation = False

    ### Locating, decompressing and loading data ###
    if len(sys.argv) > 1:
        path = sys.argv[1]+'/'
    else:
        path = 'data/'
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

    plane_db = load_data.load_data(spark, path + '*.csv', validation)

    eigenvalues, eigenvectors, pca_data = PCA.pca(plane_db.select('features_scaled', 'ArrDelay'), validation=validation)
    print(eigenvalues, eigenvectors)

    if len(sys.argv) == 1 or len(sys.argv) == 2:
        print('Generating model from data')
        correlation_matrix = data_exploration.compute_corr(plane_db.drop('features', 'features_scaled'),
                                                           plane_db.drop('features', 'features_scaled').columns,
                                                           name='Features')

        correlation_matrix_pca = data_exploration.compute_corr(pca_data.select('pca_features', 'ArrDelay'),
                                                               ['PCA_0', 'PCA_1', 'PCA_2', 'PCA_3', 'PCA_4',
                                                                'ArrDelay'], name='PCA_features')

        y_pred_lr, lr_data = models.linear_regression_model(plane_db.select('features_scaled', 'ArrDelay'))
        y_pred_dt, dt_data = models.decision_tree_model(plane_db.select('features_scaled', 'ArrDelay'))
        y_pred_lr_pca, lr_pca_data = models.linear_regression_model(pca_data, features_col="pca_features")

        complete_results = []
        complete_results.extend(lr_data)
        complete_results.extend(dt_data)
        complete_results.extend(lr_pca_data)
        with open('results.txt', 'w') as file_name:
            for line in complete_results:
                file_name.write(line+'\n')
            file_name.close()

    elif len(sys.argv) == 3:
        print('Starting test')
        model_name = sys.argv[2]
        if model_name == 'lr':
            model_path = 'lr_model'
            features = 'features_scaled'
        elif model_name == 'dt':
            model_path = 'dt_model'
            features = 'features_scaled'
        elif model_name == 'lr_pca':
            model_path = 'lr_pca_model'
            features = 'pca_features'
            plane_db = pca_data
        else:
            print('Please select a valid model')
            sys.exit()
        predictions = prediction_from_model.generate_predictions(model_path, model_name,
                                                                 plane_db.select(features, 'ArrDelay'))
