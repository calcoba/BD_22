import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler, QuantileDiscretizer, Bucketizer
from pyspark.ml import Pipeline, PipelineModel


def load_data(spark, file_path, validation=False):
    plane_data = spark.read.csv(file_path, header=True, inferSchema=True, nanValue='NA')
    
    print("Imported data: \n")
    plane_data.show(5)
    print('All csv files loaded. DataFrame created.')
    print("Number of instances:", plane_data.count())

    # Eliminate forbidden variables
    plane_data = plane_data.drop('Year', 'ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted',
                                 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay')

    # Eliminate variables that are not related with the delay
    plane_data = plane_data.drop('TailNum', 'FlightNum')

    # Eliminate Cancelled flights and, then, the cancellation columns
    plane_data = plane_data.filter(plane_data.Cancelled == 0)
    plane_data = plane_data.drop('Cancelled', 'CancellationCode', 'TailNum')


    # Numerically encode remaining categorical variables and creating new ones
    plane_data = plane_data.withColumn('Route', F.concat(plane_data.Origin, plane_data.Dest))
    indexer = [StringIndexer(inputCol=column_name, outputCol=column_name + '_index')
               for column_name in ['UniqueCarrier', 'Origin', 'Dest', 'Route']]
    bucketizer = [Bucketizer(inputCols=['CRSDepTime', 'DayofMonth', 'DayOfWeek'],
                             outputCols=['DepTimePeriod', 'MonthWeek', 'Weekend'],
                             splitsArray=[[0, 400, 800, 1200, 1600, 2000, 2400], [1, 7, 14, 21, 31], [1, 4, 7]])]
    quantilizer = [QuantileDiscretizer(inputCol='Distance', outputCol='Distance_coded', numBuckets=10)]

    pipeline = Pipeline(stages=indexer+bucketizer+quantilizer)
    plane_data = plane_data.fillna(0, subset='TaxiOut')
    plane_data = plane_data.withColumn('TotalDepDelay', plane_data.DepDelay+plane_data.TaxiOut)

    plane_data = plane_data.na.drop()
    plane_data = pipeline.fit(plane_data).transform(plane_data)
    plane_data = plane_data.withColumn('Week', F.when(plane_data.Weekend == 0, 1).otherwise(0))
    '''plane_data = plane_data.drop('Month','DayofMonth', 'DayOfWeek', 'Distance', 'UniqueCarrier_index', 'Origin_index',
                                 'Dest_index', 'Route_index', 'CRSElapsedTime', 'Month')'''
    features_to_drop = ['Month', 'DayofMonth', 'DayOfWeek', 'Distance', 'UniqueCarrier_index', 'Origin_index',
                        'Dest_index', 'Route_index', 'CRSElapsedTime', 'Month']


    # Eliminate redundant categorical columns
    cols_filtered = [c for c, t in plane_data.dtypes if t != 'string']
    plane_data_clean = plane_data.select(*cols_filtered)

    print("Preprocesed data: \n")
    plane_data_clean.show(5, False)
    print("Number of instances after preprocessing:", plane_data_clean.count())

    # Merge variables and scale them
    if not validation:
        features_to_keep = ['DepTime', 'CRSDepTime', 'CRSArrTime', 'TaxiOut', 'TotalDepDelay', 'DepTimePeriod']
        assembler = VectorAssembler(inputCols=plane_data_clean.select(*features_to_keep).columns, outputCol="features")
        scaler = StandardScaler(inputCol='features', outputCol='features_scaled')
        pipeline = Pipeline(stages=[assembler, scaler]).fit(plane_data_clean)
        data_scaled = pipeline.transform(plane_data_clean)
        pipeline.write().overwrite().save('standardization_model/')
    else:
        pipeline = PipelineModel.load('standardization_model/')
        data_scaled = pipeline.transform(plane_data_clean)


    # plane_data.select([F.count(F.when(F.isnan(c), c)).alias(c) for c in plane_data.columns]).show()
    print("Scaled and prepared data: \n")
    data_scaled.show(10, False)

    return data_scaled
