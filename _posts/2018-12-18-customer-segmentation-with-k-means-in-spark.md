---
title: Customer Segmentation with K-means in Spark
permalink: /posts/2018/12/customer-segmentation-with-k-means-in-spark
tags: [spark, machine learning, data science, k-means]
header:
  image: "/images/k-means/K-means_convergence.gif"
excerpt: "Machine Learning, Perceptron, Data Science"
mathjax: "true"
---

### Customer Segmentation
Many businesses are now facing severe competition in a highly dynamic and unstable marketing environment. In order to continue to lead the in their market, they have to provide quality products and services which are capable of responding  to changes in their customers’ needs, wishes, characteristics and behaviours. So instead of looking at all customers as one entity and engage and market to them the same way, companies should approach customers differently, depending on their needs, characteristics and behaviours. Not only does this approach increase chances of getting new customers based on the ones you have, it also helps the company with retaining the ones they already have and experimenting with different products and services to different segments of customers. They must place the same importance on strategies focused on retaining customers and bringing back former subscribers, rather than focusing only on  increasing their market share.

### K-means Clustering
In general, in order to perform customer segmentation, companies use geographical, demographical, psycho-graphical, socio-economic, behavioural characteristics and psychological attitudes toward the respective product or service. In the case of markets that feature high competition, such as the telecom market, this approach is not enough. These companies also need to consider the information related to the customers’ needs, consumer behaviour, service or payment preferences, perception of product, probability of leaving the network, growth potential and customer migration.

Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups. In simple words, the aim is to segregate groups with similar traits and assign them into clusters. The goal of the k-means algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. In the reference image below, K=2, and there are two clusters identified from the source dataset.

### Load data and EDA
```scala
// Load dataset
val customersDf = spark.read.format("csv").option("sep", ",").option("inferSchema", "true").option("header", "true").load("file://///Users/khumbokaunda/Desktop/BIGDATA/DATASETS/Telco-Customer-Churn.csv").cache()
'''
