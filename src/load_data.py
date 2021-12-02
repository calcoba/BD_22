import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


def load_data(spark, file_path):
    plane_data = spark.read.csv(file_path, header=True, inferSchema=True, nanValue='NA')
    plane_data.show(12)
    target_data = plane_data['ArrDelay']
    
    # Eliminate forbidden variables
    plane_data = plane_data.drop('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted',
                                 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay')
    print('Loaded')
    plane_data.count()
    print("Number of instances:", plane_data.count())
    
    # plane_data.filter(plane_data.CancellationCode.isNull()).show(10)
    # Eliminate Cancelled flights and, then, the cancellation columns
    plane_data = plane_data.filter(plane_data.Cancelled == 0)
    plane_data=plane_data.drop('Cancelled', 'CancellationCode')

    indexer = [StringIndexer(inputCol=column_name, outputCol=column_name + '_index').
               fit(plane_data) for column_name in ['UniqueCarrier', 'TailNum', 'Origin', 'Dest']]

    pipeline = Pipeline(stages=indexer)
    plane_data = plane_data.na.drop()
    plane_data = pipeline.fit(plane_data).transform(plane_data)
    # plane_data.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in plane_data.columns]).show()

    plane_data = plane_data.na.drop()
    plane_data.show(12)
    cols_filtered = [c for c, t in plane_data.dtypes if t != 'string']
    plane_data_clean = plane_data.select(*cols_filtered)
    plane_data_clean.show(5)
    # plane_data.select([F.count(F.when(F.isnan(c), c)).alias(c) for c in plane_data.columns]).show()
    plane_data_clean.count()
    print("Number of instances after preprocessing:", plane_data_clean.count())


    return plane_data_clean, target_data
