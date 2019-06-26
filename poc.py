# Import libraries necessary for this project
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn import metrics

sc = SparkContext("local", "SLA prediction")

spark = SparkSession.builder.getOrCreate()
# Load the POC excel in a DataFrame
data = pd.read_excel("/home/alejandro/Desktop/hadoop/data.xls")

mySchema = StructType([StructField("ID", IntegerType(), True) \
                          , StructField("TERCER_DOCUMENT", StringType(), True) \
                          , StructField("TERCER_UNIT", StringType(), True) \
                          , StructField("TEMPS_INICI", IntegerType(), True) \
                          , StructField("AGENT_ASSIGNAT", StringType(), True) \
                          , StructField("COMPLEIX_SLA", StringType(), True)])

dataset = spark.createDataFrame(data, schema=mySchema)

print(data.head(3))
print(data.size)

categoricalColumns = ["TERCER_DOCUMENT", "TERCER_UNIT", "AGENT_ASSIGNAT"]
stages = []  # stages in our Pipeline
for categoricalCol in categoricalColumns:
    # Categorical variables indexed with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="COMPLEIX_SLA", outputCol="label")
stages += [label_stringIdx]

# Transform all features into a vector using VectorAssembler
numericCols = ["TEMPS_INICI"]
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

#Confussion matrix and precission metrics
testLabel = predictions.select("prediction", "label")

mvv_list = testLabel.selectExpr("prediction as prediction", "label as label")
arr_pred = [int(row['prediction']) for row in mvv_list.collect()]
arr_label = [int(row['label']) for row in mvv_list.collect()]

tp = testLabel.rdd.map(tuple)
metricas = MulticlassMetrics(tp)
confusion_mat = metricas.confusionMatrix()
print(confusion_mat.toArray())
print("Accuracy:", metrics.accuracy_score(arr_label, arr_pred))
print("Precision:", metrics.precision_score(arr_label, arr_pred))
print("Recall:", metrics.recall_score(arr_label, arr_pred))
