import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline


def load_data(spark, file_path):
    plane_data = spark.read.csv(file_path, header=True, inferSchema=True, nanValue='NA')
    plane_data = plane_data.withColumn('Route', F.concat(plane_data.Origin, plane_data.Dest))
    plane_data.show(5)

    # Eliminate forbidden variables
    plane_data = plane_data.drop('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted',
                                 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay')
    print('All csv files loaded. DataFrame created.')
    print("Number of instances:", plane_data.count())

    # Eliminate variables that are not related with the delay
    plane_data = plane_data.drop('TaxiOut', 'TailNum', 'FlightNum', 'DepTime')

    # Eliminate Cancelled flights and, then, the cancellation columns
    plane_data = plane_data.filter(plane_data.Cancelled == 0)
    plane_data = plane_data.drop('Cancelled', 'CancellationCode', 'TailNum')

    # Numerically encode remaining categorical variables
    indexer = [StringIndexer(inputCol=column_name, outputCol=column_name + '_index').
               fit(plane_data) for column_name in ['UniqueCarrier', 'Origin', 'Dest', 'Route']]

    pipeline = Pipeline(stages=indexer)
    plane_data = plane_data.na.drop()
    plane_data = pipeline.fit(plane_data).transform(plane_data)
    # plane_data.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in plane_data.columns]).show()

    # Eliminate redundant categorical columns
    cols_filtered = [c for c, t in plane_data.dtypes if t != 'string']
    plane_data_clean = plane_data.select(*cols_filtered)
    assembler = VectorAssembler(inputCols=plane_data_clean.drop('ArrDelay').columns, outputCol="features")
    scaler = StandardScaler(inputCol='features', outputCol='features_scaled')
    pipeline = Pipeline(stages=[assembler, scaler])
    data_scaled = pipeline.fit(plane_data_clean).transform(plane_data_clean)

    # plane_data.select([F.count(F.when(F.isnan(c), c)).alias(c) for c in plane_data.columns]).show()
    data_scaled.show(5, False)
    print("Number of instances after preprocessing:", data_scaled.count())

    return data_scaled
