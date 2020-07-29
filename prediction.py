from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, FloatType

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.master", "local") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

Model = RandomForestClassificationModel.load("s3n://643-pa2/TrainingModel.model")
schema = StructType([
    StructField("\"\"\"\"\"fixed acidity\"\"\"\"", FloatType(), True),
    StructField("\"\"\"\"volatile acidity\"\"\"\"", FloatType(), True),
    StructField("\"\"\"\"citric acid\"\"\"\"", FloatType(), True),
    StructField("\"\"\"\"residual sugar\"\"\"\"", FloatType(), True),
    StructField("\"\"\"\"chlorides\"\"\"\"", FloatType(), True),
    StructField("\"\"\"\"free sulfur dioxide\"\"\"\"", FloatType(), True),
    StructField("\"\"\"\"total sulfur dioxide\"\"\"\"", FloatType(), True),
    StructField("\"\"\"\"density\"\"\"\"", FloatType(), True),
    StructField("\"\"\"\"pH\"\"\"\"", FloatType(), True),
    StructField("\"\"\"\"sulphates\"\"\"\"", FloatType(), True),
    StructField("\"\"\"\"alcohol\"\"\"\"", FloatType(), True),
    StructField("\"\"\"\"quality\"\"\"\"", FloatType(), True)
])
testing = spark.read.format("csv").option("header", "true").option("delimiter", ";").schema(schema).load("s3n://643-pa2/ValidationDataset.csv")
vectorAssembler = VectorAssembler(inputCols = ["\"\"\"\"\"fixed acidity\"\"\"\"","\"\"\"\"volatile acidity\"\"\"\"","\"\"\"\"citric acid\"\"\"\"","\"\"\"\"residual sugar\"\"\"\"","\"\"\"\"chlorides\"\"\"\"","\"\"\"\"free sulfur dioxide\"\"\"\"","\"\"\"\"total sulfur dioxide\"\"\"\"","\"\"\"\"density\"\"\"\"","\"\"\"\"pH\"\"\"\"","\"\"\"\"sulphates\"\"\"\"","\"\"\"\"alcohol\"\"\"\""], outputCol = 'features')
test_data = vectorAssembler.transform(testing)
predictions = Model.transform(test_data)
predictionAndLabels = predictions.select(['prediction', "\"\"\"\"quality\"\"\"\""]).rdd

# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)

# Overall statistics
print("F1 Score: " + str(metrics.weightedFMeasure()))