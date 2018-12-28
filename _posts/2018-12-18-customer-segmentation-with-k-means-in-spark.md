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

### HYPOTHESIS TESTING IN APACHE SPARK
Our hypothesis based on the dataset is that customers who have signed up for automatic bill payments either through credit cards, bank tranfers, and those with non-automatic payemnts through mailed checks are less likely to churn compared to customers who pay by electronic check. Before we perform hypotheis testing we have to transform our dataframe into numerical format and then into a feature vector column. Fist we use the StringIndexer transformer to index all categorical columns and the the VectorAssembler to transform our features columns to a single vector column as required for hypothesis testing and the K-means algorithm.

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
