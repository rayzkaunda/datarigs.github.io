---
title: Building and deploying a machine learning model to predict a diabetic patients hospital length of stay on AWS
tags: [aws-sagemaker-deploy]
header:
  image: "images/d-patients/d-patients.jpeg"
excerpt: "AWS Serveless"
mathjax: "true"
---

# Business Problem:
Using the given dataset, you  will develop a modern scalable full stack application on AWS cloud using a predictive model to predict how many days an admitted diabetic patient will spend 5 days or more in a hospital. We have framed our problem as a binary classification problem but it can also be a regression problem.  


### Project Overview

In 2012, there were about 36.5 million hospital stays with an average length of stay of 4.5 days
and an average cost of 10,400 dollars per stay summing up to 377 billion dollars('https://www.hcup-us.ahrq.gov/reports/statbriefs/sb180-Hospitalizations-United-States-2012.pdf').
In 2012, there were approximately 36.5 million hospital stays in the United States, representing a
hospitalization rate of 116.2 stays per 1,000 population. Across all types of stays, the average length
of a hospital stay was 4.5 days.

Predictive analytics is an increasingly important tool in the healthcare field since modern machine learning (ML) methods can use large amounts of available data to predict individual outcomes for patients. For example, ML predictions can help healthcare providers determine likelihoods of disease, aid in diagnosis, recommend treatment, and predict future wellness. For this project, I chose to focus on a more logistical metric of healthcare, hospital length-of-stay (LOS). LOS is defined as the time between hospital admission and discharge measured in days.

### Architectural Diagram

<img src="{{ site.url }}{{ site.baseurl }}/images/d-patients/serverless-frontend-sagemaker.jpeg">

# Data and Materials

In this project am going to use the Health Facts database(Cerner Corporation, Kansas City, MO), a national data
warehouse that collects comprehensive clinical records across hospitals throughout the United States. Health Facts is a voluntary program offered to organizations which use the Cerner Electronic Health Record System. The database contains data systematically collected from participating institutions electronic medical records and includes encounter data (emergency, outpatient, and inpatient), provider specialty, demographics (age, sex, and race), diagnoses and in-hospital procedures documented by ICD-9-CM codes, laboratory data, pharmacy data, in-hospital mortality, and hospital characteristics. All data were deidentified in compliance with the Health Insurance Portability and Accountability Act of 1996 before being provided to the investigators. Continuity of patient encounters within the same health system (EHR
system) is preserved.
The dataset consists of 101766 encounters of diabetic patients with 50 attributes including age, race, weight, whether a patient was readmitted  or not and length of hospital stay as well as various medications and tests administered during an encounter.

Dimension:101766*50

# Data and Materials
Transformed/Wrangled data: Comma Separated Values(csv)
* Basic setup for using SageMaker.
* Converting datasets to protobuf format used by the Amazon SageMaker algorithms and uploading to S3.
* Training SageMaker's linear learner on the data set.
* Hosting the trained model.
* Scoring using the trained model.


### Loading  Data
In order to train your model you first have to load your data into S3. After loading your data into S3 you must create a a role in IAM to give your Sagemaker instance access to the data. According to the documentation, Most Sagemaker algorithms requires that your data be in CSV format after you've done your preprocessing before uploading data to S3. Most Amazon SageMaker algorithms work best when you use the optimized protobuf recordIO format for the training data. Using this format allows you to take advantage of Pipe mode when training the algorithms that support it. In Pipe mode, your training job streams data directly from Amazon S3. Streaming can provide faster start times for training jobs and better throughput. With Pipe mode, you also reduce the size of the Amazon Elastic Block Store volumes for your training instances. It is also important that you create the S3 buckets in the same Amazon region as your notebook instance. Otherwise Amazon will throw an error saying it cannot find your data. Fortunately SageMaker provides an easy way to convert your CSV to protobuf recordIO format. Lets load the dataset and specify the bucket to load to load model artifacts.

```python
# Load dataset
import os
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()

bucket = 'datastorez'# enter your s3 bucket where you will copy data and model artifacts
prefix = 'sagemaker/Diabetic-prediction' # place to upload training files within the bucket
```
Next, we import the necessary imports for our model and visualizations.

```python
#imports

import pandas as pd
pd.set_option('display.max_columns',100)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
import json
import sagemaker.amazon.common as smac


#Let's load the data from s3.

df = pd.read_csv('s3://<bucket>/diabetic_data.csv', header = 0, sep=',')
df.head(5)

# list the columns in the df
datalist = data.columns.tolist()
# Assign new position for label column(hazardous) as required by Sagemaker's algorithms
datalist.insert(0, datalist.pop(datalist.index('hazardous')))
df = data.reindex(columns= datalist)
df.columns.tolist()

# specify columns extracted from dataset
df.columns = ['hazardous','neo_reference_id','name','absolute_magnitude','est_dia_in_kmmin','est_dia_in_kmmax','est_dia_in_mmin',
 'est_dia_in_mmax', 'est_dia_in_milesmin', 'est_dia_in_milesmax', 'est_dia_in_feetmin', 'est_dia_in_feetmax','close_approach_date',
 'epoch_date_close_approach', 'relative_velocity_km_per_sec', 'relative_velocity_km_per_hr', 'miles_per_hour', 'miss_dist.astronomical',
 'miss_dist.lunar', 'miss_dist.kilometers', 'miss_dist.miles', 'orbiting_body', 'orbit_id', 'orbit_determination_date',
 'orbit_uncertainity', 'minimum_orbit_intersection', 'jupiter_tisserand_invariant', 'epoch_osculation', 'eccentricity',
 'semi_major_axis', 'inclination', 'asc_node_longitude', 'orbital_period', 'perihelion_distance','perihelion_arg', 'aphelion_dist',
 'perihelion_time', 'mean_anomaly', 'mean_motion', 'equinox']
 ```
 For the sake of this tutorial we drop some columns containing dates but if your model requires you have to preprocess your dates to strings.

```python
df_clean = df.drop(['close_approach_date', 'orbiting_body', 'orbit_determination_date','epoch_date_close_approach', 'orbit_determination_date', 'equinox'], axis=1)
# print the shape of the data file
print(df_clean.shape)
```
(4687, 35)

### Data Prepocessing and Cleanup
---
I used to hate cleaning and wrangling data but now it's the love of my life. I simply love it. I know many of you don't believe me. Dirty data is the norm and as a former machinist I love to get my hands dirty without the oil smell. The key to cleaning data is simple. Love it. To have a sucessful data project one must clean and explore the data intensively in order to ask and answer the right questions.

The data contains some columns with missing values and non-numeric columns and will need alot of clean up and prepocessing before feeding it to our model.

```python
# Make plots
plt.rcParams['figure.figsize'] = (20, 9)
sns.distplot(df_clean['absolute_magnitude'])
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/d-patients/d_patients_data_cleaning.gif" alt="data cleaning">
### Observation:
This histogram is a normal distribution as it is bell shaped.

<img src="{{ site.url }}{{ site.baseurl }}/images/sagemaker/sagemaker2.jpg" alt="sagemaker training">

### Observation:
Non-Normality – Histogram: a right-skewed distribution, plotted as a histogram. The histogram is not bell-shaped, indicating that the distribution is not normal.

```python
sns.distplot(df_clean['orbit_uncertainity'])
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/sagemaker/sagemaker3.jpg" alt="sagemaker training">

### Observation:
This is a bi-model distribution and the data can reveal a shift in the process.For Processes that display this distribution, it is normally understood that there are 2 independent sources of Variation that result in Peaks within the data.

Correlation between data:
Are any of the columns correlated?
```python
cmap = sns.diverging_palette(0, 255, sep=1, n=256, as_cmap=True)
correlations = df_clean[['hazardous','neo_reference_id','name','absolute_magnitude','est_dia_in_kmmin','est_dia_in_kmmax','est_dia_in_mmin',
 'est_dia_in_mmax', 'est_dia_in_milesmin', 'est_dia_in_milesmax', 'est_dia_in_feetmin', 'est_dia_in_feetmax',
 'relative_velocity_km_per_sec', 'relative_velocity_km_per_hr', 'miles_per_hour', 'miss_dist.astronomical',
 'miss_dist.lunar', 'miss_dist.kilometers', 'miss_dist.miles', 'orbit_id',
 'orbit_uncertainity', 'minimum_orbit_intersection', 'jupiter_tisserand_invariant', 'epoch_osculation', 'eccentricity',
 'semi_major_axis', 'inclination', 'asc_node_longitude', 'orbital_period', 'perihelion_distance','perihelion_arg', 'aphelion_dist',
 'perihelion_time', 'mean_anomaly', 'mean_motion']].corr()
sns.heatmap(correlations, cmap=cmap)
plt.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/sagemaker/sagemaker4.jpg" alt="sagemaker training">

Next, we save the data to notebook instance, create  features and label and convert to protobuf recordIO format.

```python
# save the data
df.to_csv("data.csv", sep=',', index=False)

# print the shape of the data file
print(df_clean.shape)

# show the top few rows
display(df_clean.head())

# describe the data object
display(df_clean.describe())

# we will also summarize the categorical field hazardous
display(df_clean.hazardous.value_counts())
```
<img src="{{ site.url }}{{ site.baseurl }}/images/sagemaker/sagemaker5.jpg" alt="sagemaker training">

## Create Features and Labels
#### Split the data into 80% training, 10% validation and 10% testing.
```python
rand_split = np.random.rand(len(df_clean))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
test_list = rand_split >= 0.9

data_train = df_clean[train_list]
data_val = df_clean[val_list]
data_test = df_clean[test_list]

val_y = data_val.iloc[:,0].astype(int).as_matrix();
val_X = data_val.iloc[1:3,:].as_matrix();

train_y = data_train.iloc[:,0].astype(int).as_matrix();
train_X = data_train.iloc[:,3:].as_matrix();

val_y = data_val.iloc[:,0].astype(int).as_matrix();
val_X = data_val.iloc[:,3:].as_matrix();

test_y = data_test.iloc[:,0].astype(int).as_matrix();
test_X = data_test.iloc[:,3:].as_matrix();
```

Now, we'll convert the datasets to the recordIO-wrapped protobuf format used by the Amazon SageMaker algorithms, and then upload this data to S3. We'll start with training data.
```python
train_file = 'linear_train.data'

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, train_X.astype('float32'), train_y.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_fileobj(f);

validation_file = 'linear_validation.data'

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, val_X.astype('float32'), val_y.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation', validation_file)).upload_fileobj(f)
```
---
## Train

Now we can begin to specify our linear model.  Amazon SageMaker's Linear Learner actually fits many models in parallel, each with slightly different hyperparameters, and then returns the one with the best fit.  This functionality is automatically enabled.  We can influence this using parameters like:

- `num_models` to increase to total number of models run.  The specified parameters will always be one of those models, but the algorithm also chooses models with nearby parameter values in order to find a solution nearby that may be more optimal.  In this case, we're going to use the max of 32.
- `loss` which controls how we penalize mistakes in our model estimates.  For this case, let's use absolute loss as we haven't spent much time cleaning the data, and absolute loss will be less sensitive to outliers.
- `wd` or `l1` which control regularization.  Regularization can prevent model overfitting by preventing our estimates from becoming too finely tuned to the training data, which can actually hurt generalizability.  In this case, we'll leave these parameters as their default "auto" though.

```python
# See 'Algorithms Provided by Amazon SageMaker: Common Parameters' in the SageMaker documentation for an explanation of these values.
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'linear-learner')

linear_job = 'ASTEROID-linear-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())



print("Job name is:", linear_job)

linear_training_params = {
    "RoleArn": role,
    "TrainingJobName": linear_job,
    "AlgorithmSpecification": {
        "TrainingImage": container,
        "TrainingInputMode": "File"
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.c4.2xlarge",
        "VolumeSizeInGB": 10
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/train/".format(bucket, prefix),
                    "S3DataDistributionType": "ShardedByS3Key"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/validation/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        }

    ],
    "OutputDataConfig": {
        "S3OutputPath": "s3://{}/{}/".format(bucket, prefix)
    },
    "HyperParameters": {
        "feature_dim": "32",
        "mini_batch_size": "100",
        "predictor_type": "regressor",
        "epochs": "10",
        "num_models": "32",
        "loss": "absolute_loss"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 60 * 60
    }
}
```
Now let's kick off our training job in SageMaker's distributed, managed training, using the parameters we just created. Because training is managed, we don't have to wait for our job to finish to continue, but for this case, let's use boto3's 'training_job_completed_or_stopped' waiter so we can ensure that the job has been started.
```python
%%time

region = boto3.Session().region_name
sm = boto3.client('sagemaker')

sm.create_training_job(**linear_training_params)

status = sm.describe_training_job(TrainingJobName=linear_job)['TrainingJobStatus']
print(status)
sm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=linear_job)
if status == 'Failed':
    message = sm.describe_training_job(TrainingJobName=linear_job)['FailureReason']
    print('Training failed with the following error: {}'.format(message))
    raise Exception('Training job failed')
```
InProgress
CPU times: user 68.6 ms, sys: 1.06 ms, total: 69.7 ms
Wall time: 4min

---
## Host

Now that we've trained the linear algorithm on our data, let's setup a model which can later be hosted.  We will:
1. Point to the scoring container
1. Point to the model.tar.gz that came from training
1. Create the hosting model

```python
linear_hosting_container = {
    'Image': container,
    'ModelDataUrl': sm.describe_training_job(TrainingJobName=linear_job)['ModelArtifacts']['S3ModelArtifacts']
}

create_model_response = sm.create_model(
    ModelName=linear_job,
    ExecutionRoleArn=role,
    PrimaryContainer=linear_hosting_container)

print(create_model_response['ModelArn'])
```
Once we've setup a model, we can configure what our hosting endpoints should be.  Here we specify:
1. EC2 instance type to use for hosting
1. Initial number of instances
1. Our hosting model name

```python
linear_endpoint_config = 'DEMO-linear-endpoint-config-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print(linear_endpoint_config)
create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName=linear_endpoint_config,
    ProductionVariants=[{
        'InstanceType': 'ml.m4.xlarge',
        'InitialInstanceCount': 1,
        'ModelName': linear_job,
        'VariantName': 'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])
```
Now that we've specified how our endpoint should be configured, we can create them. This can be done in the background, but for now let's run a loop that updates us on the status of the endpoints so that we know when they are ready for use.

```python
%%time

linear_endpoint = 'DEMO-linear-endpoint-' + time.strftime("%Y%m%d%H%M", time.gmtime())
print(linear_endpoint)
create_endpoint_response = sm.create_endpoint(
    EndpointName=linear_endpoint,
    EndpointConfigName=linear_endpoint_config)
print(create_endpoint_response['EndpointArn'])

resp = sm.describe_endpoint(EndpointName=linear_endpoint)
status = resp['EndpointStatus']
print("Status: " + status)

sm.get_waiter('endpoint_in_service').wait(EndpointName=linear_endpoint)

resp = sm.describe_endpoint(EndpointName=linear_endpoint)
status = resp['EndpointStatus']
print("Arn: " + resp['EndpointArn'])
print("Status: " + status)

if status != 'InService':
    raise Exception('Endpoint creation did not succeed')
```
## Predict
### Predict on Test Data

Now that we have our hosted endpoint, we can generate statistical predictions from it.  Let's predict on our test dataset to understand how accurate our model is.

There are many metrics to measure classification accuracy.  Common examples include include:
- Precision
- Recall
- F1 measure
- Area under the ROC curve - AUC
- Total Classification Accuracy
- Mean Absolute Error

For our example, we'll keep things simple and use total classification accuracy as our metric of choice. We will also evaluate  Mean Absolute  Error (MAE) as the linear-learner has been optimized using this metric, not necessarily because it is a relevant metric from an application point of view. We'll compare the performance of the linear-learner against a naive benchmark prediction which uses majority class observed in the training data set for prediction on the test data.

```python
 #function to convert an array to a csv
def np2csv(arr):
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()    


#next, we'll invoke the endpoint to get predictions.
runtime= boto3.client('runtime.sagemaker')

payload = np2csv(test_X)
response = runtime.invoke_endpoint(EndpointName=linear_endpoint,
                                   ContentType='text/csv',
                                   Body=payload)
result = json.loads(response['Body'].read().decode())
test_pred = np.array([r['score'] for r in result['predictions']])

#Let's compare linear learner based mean absolute prediction errors from a baseline prediction which uses majority class to predict every instance.
test_mae_linear = np.mean(np.abs(test_y - test_pred))
test_mae_baseline = np.mean(np.abs(test_y - np.median(train_y))) ## training median as baseline predictor

print("Test MAE Baseline :", round(test_mae_baseline, 3))
print("Test MAE Linear:", round(test_mae_linear,3))
```
Test MAE Baseline : 0.18
Test MAE Linear: 0.181
```python
#Let's compare predictive accuracy using a classification threshold of 0.5 for the predicted and compare against the majority class prediction from training data set
test_pred_class = (test_pred > 0.5)+0;
test_pred_baseline = np.repeat(np.median(train_y), len(test_y))

prediction_accuracy = np.mean((test_y == test_pred_class))*100
baseline_accuracy = np.mean((test_y == test_pred_baseline))*100

print("Prediction Accuracy:", round(prediction_accuracy,1), "%")
print("Baseline Accuracy:", round(baseline_accuracy,1), "%")
```
Our accuracy on the model

Prediction Accuracy: 82.0 %
Baseline Accuracy: 82.0 %  

Finally, run the following to delete the endpoint instance to avoid charges and stop the notebook instance(if you want to keep the notebook don't delete the notebook instance) in the Sagemaker console(This does not delete the notebook).
```python
sm.delete_endpoint(EndpointName=linear_endpoint)
```
### Conclusion
In this post,  we learned the basic foundations loading data, training, evaluation and deploying a model in AWS SageMaker. The key takeways.
- Data preprocessing: The data in the feature  and label columns have to be float32 format to convert to recordIO-wrapped protobuf format. The label column has be the     first in the dataframe as expected by most SageMaker algorithms.
