from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('big_data_project').getOrCreate()


def load_data(file_path):
    plane_data = spark.read.csv(file_path, header=True)
    plane_data.show(5)
    target_data = plane_data['ArrTime']
    plane_data = plane_data.drop('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay',
                                 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay')
    print('Loaded')
    plane_data.show(5)
    return plane_data, target_data


file_paths = ['C:/Users/carlo/PycharmProjects/pythonProject/data/2006.csv']

plane_db, target_db = load_data(file_paths)
