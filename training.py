from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.master", "local") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
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
training = spark.read.format("csv").option("header", "true").option("delimiter", ";").schema(schema).load("s3n://643-pa2/TrainingDataset.csv")
vectorAssembler = VectorAssembler(inputCols = ["\"\"\"\"\"fixed acidity\"\"\"\"","\"\"\"\"volatile acidity\"\"\"\"","\"\"\"\"citric acid\"\"\"\"","\"\"\"\"residual sugar\"\"\"\"","\"\"\"\"chlorides\"\"\"\"","\"\"\"\"free sulfur dioxide\"\"\"\"","\"\"\"\"total sulfur dioxide\"\"\"\"","\"\"\"\"density\"\"\"\"","\"\"\"\"pH\"\"\"\"","\"\"\"\"sulphates\"\"\"\"","\"\"\"\"alcohol\"\"\"\""], outputCol = 'features')
training_data = vectorAssembler.transform(training)
training_data = training_data.select(['features', "\"\"\"\"quality\"\"\"\""])
training_data.show(3)
rf = RandomForestClassifier(labelCol="\"\"\"\"quality\"\"\"\"",
                            featuresCol='features',
                            maxDepth=10)
#lr = LinearRegression(featuresCol = 'features', labelCol="\"\"\"\"quality\"\"\"\"", maxIter=10, regParam=0.3, elasticNetParam=0.8)
model = rf.fit(training_data)
model.save("s3n://643-pa2/TrainingModel.model")


