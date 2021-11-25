import src.load_data as load_data
from pyspark.sql import SparkSession


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spark = SparkSession.builder.appName('big_data_project').getOrCreate()
    path = 'data/'
    files = ['2006']
    for file_name in files:
        plane_db, target_db = load_data.load_data(spark, path+file_name+'.csv')

    print(plane_db.dtypes)
    from pyspark.ml.stat import Correlation
    from pyspark.ml.feature import VectorAssembler

    # convert to vector column first
    vector_col = "corr_features"
    plane_int_db = plane_db['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'FlightNum',
                            'Distance']
    assembler = VectorAssembler(inputCols=plane_int_db.columns, outputCol=vector_col)
    df_vector = assembler.transform(plane_int_db).select(vector_col)

    # get correlation matrix
    matrix = Correlation.corr(df_vector, vector_col)
    print(matrix.collect()[0]["pearson({})".format(vector_col)].values)
