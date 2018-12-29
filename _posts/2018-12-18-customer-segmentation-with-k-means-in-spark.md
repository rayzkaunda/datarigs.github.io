---
title: Customer Segmentation with K-means in Spark
tags: [k-means]
header:
  image: "images/k-means/customer-segment.jpg"
excerpt: "Machine Learning, Perceptron, Data Science"
mathjax: "true"
---

### Customer Segmentation
Many businesses are now facing severe competition in a highly dynamic and unstable marketing environment. In order to continue to lead the in their market, they have to provide quality products and services which are capable of responding  to changes in their customers’ needs, wishes, characteristics and behaviours. So instead of looking at all customers as one entity and engage and market to them the same way, companies should approach customers differently, depending on their needs, characteristics and behaviours. Not only does this approach increase chances of getting new customers based on the ones you have, it also helps the company with retaining the ones they already have and experimenting with different products and services to different segments of customers. They must place the same importance on strategies focused on retaining customers and bringing back former subscribers, rather than focusing only on  increasing their market share.

### K-means Clustering
In general, in order to perform customer segmentation, companies use geographical, demographical, psycho-graphical, socio-economic, behavioural characteristics and psychological attitudes toward the respective product or service. In the case of markets that feature high competition, such as the telecom market, this approach is not enough. These companies also need to consider the information related to the customers’ needs, consumer behaviour, service or payment preferences, perception of product, probability of leaving the network, growth potential and customer migration.

Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups. In simple words, the aim is to segregate groups with similar traits and assign them into clusters. The goal of the k-means algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. In the reference image below, K=2, and there are two clusters identified from the source dataset.

### The Dataset and Define Problem

Context
“Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs.”

The dataset is a csv file with real world customer information for a cell and internet company. Each row represents a customer, each column contains customer’s attributes described on the column Metadata.The data set includes information about:

- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

### How to run machine learning algorithms in Apache Spark with Zeppellin and Scala.

For this post, I'll show you how to run the K-means in Apache Spark using the mlib library with the scala api and Zeppellin notebook. I personally like to work with scala when working with spark but you can also run the notebook using the python and R api. For optimal performance it is good practice to use dataframes than RDDs.

### Load data and EDA

```scala
// Load imports
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import spark.implicits._

// Load dataset
val customersDf = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header", "true").load("file://///Users/khumbokaunda/Desktop/BIGDATA/DATASETS/Telco-Customer-Churn.csv").cache()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/k-means/k-means1.jpg" alt="dataframe">

Now lets register the dataframe to use Spark SQL queries for visualizations. With the Zeppelin Notebook, you can display query results in table or chart formats. Here are some example Spark SQL queries on the customers dataset.
### What is the nature of payment methods in relation to churn?
```scala
customersDf.createOrReplaceTempView("customersDf")
```
<img src="{{ site.url }}{{ site.baseurl }}/images/k-means/k-means2.jpg" alt="dataframe">

In the pie chart we can see the different payment methods and the percentage of customer churn for each. For example, for credit card and automatic payments the churn was 2.27% and for bank tranfers it was 4.41%. Furthermore, the churn for mailed checks was 4.67% and electronic checks was 13.22%. From this we can form a hypothesis of our dataset and before performing our analysis.

### TRAIN K-MEANS
Fist we use the StringIndexer transformer to index all categorical columns and the the VectorAssembler to transform our features columns to a single vector column as required for hypothesis testing and the K-means algorithm.

```scala
//Index categorical columns
 val featureCol = customersDf.columns
   val indexers = featureCol.map { colName =>
   new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed")
   }


//Define pipeline and transform dataset
   val pipeline = new Pipeline().setStages(indexers)      
   val newDF = pipeline.fit(customersDf).transform(customersDf)
   newDF.show(5)

// Select only the indexed columns
val assembler = new VectorAssembler().setInputCols(Array("customerID_indexed", "gender_indexed", "seniorCitizen_indexed", "Partner_indexed", "Dependents_indexed", "tenure_indexed", "PhoneService_indexed", "MultipleLines_indexed", "InternetService_indexed", "OnlineSecurity_indexed" , "OnlineBackup_indexed",
"DeviceProtection_indexed", "TechSupport_indexed", "StreamingTV_indexed", "Contract_indexed", "PaperlessBilling_indexed", "PaymentMethod_indexed", "MonthlyCharges_indexed", "TotalCharges_indexed")).setOutputCol("features")

val output = assembler.transform(featureDf)
output.show(5)   
```
<img src="{{ site.url }}{{ site.baseurl }}/images/k-means/k-means3.jpg" alt="dataframe">


```scala
val df = output.select("features", "Churn_indexed").withColumnRenamed("Churn_indexed", "label")
//split data set to train, test
val Array(train, test) = df.randomSplit(Array(0.7,0.3))
```
Parameter selection
The best choice of k depends upon the data; generally, larger values of k reduces effect of the noise on the classification, but make boundaries between classes less distinct. A good k can be selected by various heuristic techniques. The special case where the class is predicted to be the class of the closest training sample (i.e. when k = 1) is called the nearest neighbor algorithm.

The accuracy of the k-NN algorithm can be severely degraded by the presence of noisy or irrelevant features, or if the feature scales are not consistent with their importance. Much research effort has been put into selecting or scaling features to improve classification. A particularly popular approach is the use of evolutionary algorithms to optimize feature scaling. Another popular approach is to scale features by the mutual information of the training data with the training classes.

In binary (two class) classification problems, it is helpful to choose k to be an odd number as this avoids tied votes. One popular way of choosing the empirically optimal k in this setting is via bootstrap method.


```scala
// Trains a k-means model.

val kmeans = new KMeans().setK(5).setSeed(1L).setFeaturesCol("features").setPredictionCol("prediction")
val model = kmeans.fit(train)

// Make predictions
val segments = model.transform(test)

//here the prediction column shows the cluster
segments.show(5)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/k-means/k-means4.jpg" alt="dataframe">

Now lets evaluate clustering by computing Silhouette score

```scala
val evaluator = new ClusteringEvaluator()

val silhouette = evaluator.evaluate(segments)
println(s"Silhouette with squared euclidean distance = $silhouette")
```
<img src="{{ site.url }}{{ site.baseurl }}/images/k-means/k-means6.jpg" alt="dataframe">

Silhouette refers to a method of interpretation and validation of consistency within clusters of data. The technique provides a succinct graphical representation of how well each object lies within its cluster.

The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.
The silhouette can be calculated with any distance metric, such as the Euclidean distance or the Manhattan distance.

### Analyzing Results

<img src="{{ site.url }}{{ site.baseurl }}/images/k-means/k-means6.jpg" alt="dataframe">

For a better analysis we need to join the original dataframe with the results.

```scala
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

val resultsDF = featureDf.join(segments, featureDf.col("Churn_indexed") === segments.col("Churn_indexed"))

// join the three datasets(original, indexed and k-means results) for better analysis
// first we change the customer_id column back to string type in the indexed dataframe(featureDf) then join on customer_id
val converter = new IndexToString().setInputCol("customerID_indexed").setOutputCol("customerID")

val converted = converter.transform(featureDf)

// join first 2 dataframes
val resultsData = customersDf.join(segments, customersDf.col("customerID") ===customersDf.col("customerID"), "cross")

// join the 3 datasets(original, indexed and k-means results) for better analysis

val finalDataset = resultsData.join(converted, resultsData.col("customerID") ===resultsData.col("customerID"), "cross")
finalDataset.show(5)
```
<img src="{{ site.url }}{{ site.baseurl }}/images/k-means/k-means6.jpg" alt="dataframe">

```scala
// our 3 dataframes have been joined and then create a temp view for SQL queries
finalDataset.createOrReplaceTempView("finalDataset")
```
