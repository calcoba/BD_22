import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler, QuantileDiscretizer, Bucketizer
from pyspark.ml import Pipeline


def load_data(spark, file_path):
    plane_data = spark.read.csv(file_path, header=True, inferSchema=True, nanValue='NA')
    
    print("Imported data: \n")
    plane_data.show(5)
    print('All csv files loaded. DataFrame created.')
    print("Number of instances:", plane_data.count())

    # Eliminate forbidden variables
    plane_data = plane_data.drop('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted',
                                 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay')

    # Eliminate variables that are not related with the delay
    plane_data = plane_data.drop('TailNum', 'DayofMonth', 'FlightNum')

    # Eliminate Cancelled flights and, then, the cancellation columns
    plane_data = plane_data.filter(plane_data.Cancelled == 0)
    plane_data = plane_data.drop('Cancelled', 'CancellationCode')

    # Numerically encode remaining categorical variables and creating new ones
    plane_data = plane_data.withColumn('Route', F.concat(plane_data.Origin, plane_data.Dest))
    indexer = [StringIndexer(inputCol=column_name, outputCol=column_name + '_index')
               for column_name in ['UniqueCarrier', 'Origin', 'Dest', 'Route']]
    bucketizer = [Bucketizer(inputCols=['CRSDepTime', 'Month', 'DayOfWeek'],
                             outputCols=['DepTimePeriod', 'Season', 'WeekPeriod'],
                             splitsArray=[[0, 600, 1200, 1800, 2400], [1, 3, 6, 9, 12], [1, 5, 7]])]

    pipeline = Pipeline(stages=indexer+bucketizer)
    plane_data = plane_data.fillna(0, subset='TaxiOut')
    plane_data = plane_data.na.drop()
    plane_data = pipeline.fit(plane_data).transform(plane_data)
    plane_data = plane_data.drop('Month', 'DayOfWeek', 'Distance', 'UniqueCarrier_index', 'Origin_index',
                                 'Dest_index', 'Route_index')

    # Eliminate redundant categorical columns
    cols_filtered = [c for c, t in plane_data.dtypes if t != 'string']
    plane_data_clean = plane_data.select(*cols_filtered)

    print("Preprocesed data: \n")
    plane_data_clean.show(5, False)
    print("Number of instances after preprocessing:", plane_data_clean.count())

    # Merge variables and scale them
    assembler = VectorAssembler(inputCols=plane_data_clean.drop('ArrDelay').columns, outputCol="features")
    scaler = StandardScaler(inputCol='features', outputCol='features_scaled')
    pipeline = Pipeline(stages=[assembler, scaler])
    data_scaled = pipeline.fit(plane_data_clean).transform(plane_data_clean)

    return data_scaled
