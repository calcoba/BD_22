import src.load_data as load_data
from src import data_exploration
from pyspark.sql import SparkSession

if __name__ == '__main__':
    spark = SparkSession.builder.appName('big_data_project').getOrCreate()
    path = 'data/'
    files = ['2006']
    for file_name in files:
        plane_db, target_db = load_data.load_data(spark, path+file_name+'.csv')

    data_exploration.compute_corr(plane_db)

