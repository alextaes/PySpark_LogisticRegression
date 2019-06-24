# Import libraries necessary for this project
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col

sc = SparkContext("local", "SLA prediction")

spark = SparkSession.builder.getOrCreate()
# Load the POC excel in a DataFrame
data = pd.read_excel("/home/Desktop/hadoop/data.xls")

mySchema = StructType([StructField("id", IntegerType(), True) \
                          , StructField("indep_variable_1", StringType(), True) \
                          , StructField("indep_variable_2", StringType(), True) \
                          , StructField("indep_variable_3", IntegerType(), True) \
                          , StructField("indep_variable_4", StringType(), True) \
                          , StructField("dep_variable", StringType(), True)])

dataset = spark.createDataFrame(data, schema=mySchema)

print(data.head(3))
print(data.size)

categoricalColumns = ["indep_variable_1", "indep_variable_2", "indep_variable_4"]
stages = []  # stages in our Pipeline
for categoricalCol in categoricalColumns:
    # Variables categoricas indexadas con StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="dep_variable", outputCol="label")
stages += [label_stringIdx]

# Transform all features into a vector using VectorAssembler
numericCols = ["indep_variable_3"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(dataset)
preppedDataDF = pipelineModel.transform(dataset)

# Fit model to prepped data
lrModel = LogisticRegression().fit(preppedDataDF)

# Keep relevant columns
selectedcols = ["label", "features"]
dataset = preppedDataDF.select(selectedcols)

print(dataset.head(5))
dataset.show()

# Randomly split data into training and test sets. set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())

# Create initial LogisticRegression model & Train model with Training Data
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
lrModel2 = lr.fit(trainingData)

# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel2.transform(testData)
print(type(predictions))
print(list(predictions))
# View model's predictions and probabilities of each prediction class
selected = predictions.select("features", "label", "probability", "prediction")
selected.orderBy(col("label").desc()).show()

