---
title: Part 2:Building and deploying a machine learning model to predict a diabetic patients hospital length of stay on AWS
tags: [aws-sagemaker-serverless]
header:
  image: "images/d-patients/d-patients.jpeg"
excerpt: "AWS Serveless"
mathjax: "true"
---

# Introduction:
In part 1 of the post we trained processed the data, trained our model, evaluated the results. In this post we are going to use the same model to deploy the  endpoint to serve real time predictions on new data. You can look at part one of the post here(link).

# Architectural Diagram

<img src="{{ site.url }}{{ site.baseurl }}/images/d-patients/serverless-frontend-sagemaker.jpeg">

# Workflow Summary
* A serverless front end hosted on S3 through API Gateway will collect the patients data.  
* The data is then will invoke a Lambda function which will call the Sagemaker endpoint to provide a prediction.
* Through the same Lambda function Sagemaker provides the output prediction to the client application.

### Create a model
I already created a trained my model and saved the artifacts to an S3 bucket.In the AWS console go to Sagemaker select models and chose the model you trained. in my case I'll select my pre-trained model. Next you click on create endpoint.

<img src="{{ site.url }}{{ site.baseurl }}/images/d-patients/serverless-frontend-sagemaker.jpeg">

Set the Model name to los-model for this example, and then under IAM role, select Create a new role.

This role doesn’t need any special access to Amazon S3, so we can select None under S3 buckets you specify, and then choose Create role. We would need these permissions if we were building or training a model, but since we are using a pre-built model here, it is not necessary.


Now enter the following values for the Primary container:

Location of inference code image: 305705277353.dkr.ecr.us-east-1.amazonaws.com/decision-trees-sample:latest

Location of model artifacts: s3://aws-machine-learning-blog/artifacts/decision-trees/model.tar.gz

These locations are for the pre-built model and Docker container that we are using for this example. Choose Create model. This allows Amazon SageMaker to know how to make inferences, and where to find our specific model. In the future, you can use your own model.

Create an endpoint configuration
While still in the Amazon SageMaker console, in the left navigation pane, under Inference, select Endpoint configuration. Now choose the Create endpoint configuration button. Name the Endpoint configuration name “decision-trees”, and then choose Add model at the bottom of the New endpoint configuration block.

In this new Add model dialogue box, select the decision-trees model that we created in the previous step, and then choose Save.

Your new endpoint configuration should look like this:

<img src="{{ site.url }}{{ site.baseurl }}/images/d-patients/serverless-frontend-sagemaker.jpeg">

Select the Create endpoint configuration button. An endpoint configuration lets Amazon SageMaker know what model to use, from the previous section, what kind of instance to use, and how many instances to initialize the endpoint with. In the next section, we’ll create the actual endpoint, which will spin up the instance.

Create an endpoint
While still in the SageMaker console, in the left navigation pane, under Inference, select Endpoints. Choose the Create endpoint button. Under Endpoint name, enter “decision-trees”. Under Attach endpoint configuration, leave the default value of Use an existing endpoint configuration.

Under Endpoint configuration, select the decision-trees endpoint we created in the last step, and then choose the Select endpoint configuration button. The result should look like this:

<img src="{{ site.url }}{{ site.baseurl }}/images/d-patients/serverless-frontend-sagemaker.jpeg">

Now choose the Create endpoint button.

You’ve now deployed our pre-built Scikit model based on the Iris dataset! We can move on to building a serverless frontend for our endpoint.

Warning: Leaving a SageMaker endpoint running will cost you money. If you are following this blog as a learning experience, don’t forget to delete your endpoint when you’re done, so that you don’t incur further charges.

### Create a serverless API action using Chalice
Now that our SageMaker endpoint is available, we need to create an API action that can access the endpoint in order to produce results that can be served to an end user. For this blog post, we use the Chalice framework to deploy a simple Flask-like application onto API Gateway to trigger a Lambda function which will interact with our SageMaker endpoint.

Chalice is a serverless microframework for AWS. It allows you to quickly create and deploy applications that use Amazon API Gateway and AWS Lambda. Because our current endpoint expects input in CSV format, we need to do some preprocessing to transform HTML form data into a CSV file that the endpoint is expecting. Chalice lets us do this quickly and efficiently, compared to building your own Lambda function.

If you have more questions about Chalice’s advantages and disadvantages, I recommend visiting the Chalice GitHub repository.

Development environment
So that we have a consistent environment, let’s use an Amazon EC2 instance for our development environment. On the AWS Management Console, select Services, and then, under Compute, select EC2.  Now choose the Launch Instance button.  The Amazon Linux AMI comes with most of the development tools we will need, so it will be a good environment for us. Choose Select.

<img src="{{ site.url }}{{ site.baseurl }}/images/d-patients/serverless-frontend-sagemaker.jpeg">
We won’t be doing anything resource intensive, so I recommend selecting a t2.micro, and then choosing the Review and Launch button. Finally choose the Launch button one more time.

You will be prompted to Select an existing key pair or create a new key pair. Choose whichever is most convenient for you to be able to connect to your instance. If you are unfamiliar with connecting to an EC2 instance, instructions can be found in the documentation at Connect to Your Linux Instance.

Before you connect to your instance, you should give the instance some permissions, so that you don’t need to use any credentials to deploy your Chalice application. On the AWS Management Console, go to Services, and then, under Security, Identity & Compliance, select IAM. Select Roles on the left, and then choose Create role.

For Select type of trusted entity, select AWS service and then EC2. Under Choose the service that will use this role, select EC2 (Allows EC2 instances to call AWS services on your behalf). Choose Next: Permissions.

On this screen, select the following permissions: AmazonAPIGatewayAdministrator, AWSLambdaFullAccess, AmazonS3FullAccess and IAMFullAccess, before selecting Next: Review.

For Role name, type chalice-dev, and type a description similar to “Allows an EC2 instance to deploy a Chalice application.” Choose Create role.

Now we need to attach our new role to our running EC2 instance.  Go back to the EC2 console by selecting Services, and then, under Compute, select EC2. Select Running instances.  Select the instance you launched earlier, and choose Actions, Instance Settings, and then choose Attach/Replace IAM role.

Under IAM role, select “chalice-dev” and then choose Apply.

Now you can go ahead and connect to your EC2 instance.

Setting up Chalice
After you are connected to your EC2 instance, you need to install Chalice and the AWS SDK for Python (Boto3). Issue the following command:

```python
sudo pip install chalice boto3
```
To make sure that our application is deployed in the same Region as our model, we’ll set an environmental variable with the following command:
```python
export AWS_DEFAULT_REGION=us-east-1
```
With Chalice now installed, we can create our new chalice project. Let’s download a sample application, and change into the project directory. You can do this with the following commands:
```python
wget https://s3.amazonaws.com/aws-machine-learning-blog/artifacts/decision-trees/decision-trees.tgz
tar xzvf decision-trees.tgz --warning=no-unknown-keyword
cd decision-trees
```
The app.py file in this package is specifically designed to interact with the pre-built model we deployed earlier. We also downloaded a requirements.txt file, to let Chalice know what dependencies our frontend will require, and some additional hidden configuration files in the “.chalice” folder, which help to manage policy permissions.

Let’s take a quick look at the source, to try to get an idea of how the app looks. Run the following command:
```python
cat app.py
```
Now that you have the necessary Chalice project files, you can deploy the application. To do this, run the following command from your terminal:
```python
chalice deploy
```
After this is over, it will return a URI for your chalice endpoint. Save the Rest API URL for later because we will need to put it in our HTML file for our frontend.

You have now deployed a Lambda function attached to an API Gateway endpoint, which can talk to your SageMaker endpoint. All you need now is an HTML frontend to post data to your API Gateway. When a user submits a request using the frontend application, it goes to the API Gateway. This triggers the Lambda function, which executes based on the app.py file included in the Chalice application and sends data to the SageMaker endpoint you’ve created.  Any necessary preprocessing can be done in a custom app.py file.

### Generate an HTML user interface
We now have our model hosted in SageMaker, and an API Gateway interface for interacting with our endpoint. We still don’t have a proper user interface to make it possible for a user to submit new data to our model and generate live predictions. Fortunately, all we need is to serve a simple HTML form that will POST our data to our Chalice application’s endpoint.

Amazon S3 will make this easy for us.

Let’s use the command line tools to create a website bucket on Amazon S3. Select a bucket name and run the following commands:
```python
aws s3api create-bucket --bucket <bucket name> --region us-east-1

aws s3 website s3://<bucket name>/ --index-document index.html --error-document error.html
```
Now we need to upload a sample HTML file to our bucket to serve as our frontend. Let’s download a sample HTML file and edit it for our purposes. The chalice endpoint here is the URI we saved above from the deploy command.

```python
wget https://s3.amazonaws.com/aws-machine-learning-blog/artifacts/decision-trees/index.html
sed -i s@CHALICE_ENDPOINT@<your rest api URL>@g index.html
```
Let’s take a look at the index.html.
```python
cat index.html
```
The important part of this file is the action on the form that points to your API Gateway endpoint. This allows the HTML file to POST the uploaded file to our Lambda function, which will communicate with our SageMaker endpoint.

Now that we have created our HTML frontend, we need to upload it to our Amazon S3 website. Run the following command:
```python
aws s3 cp index.html s3://<bucket name>/index.html --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers
```
Your new user interface will be available at a url similar to: http://<bucket_name>.s3.amazonaws.com/index.html

<img src="{{ site.url }}{{ site.baseurl }}/images/d-patients/Diamed.jpeg" alt="Diamed.jpeg">

# Conclusion
Congratulations! You now have a fully functional serverless frontend application for the model that you built, trained, and hosted on Amazon SageMaker!  Using this address, you can now have users submit new data to your model and produce live predictions on the fly.

The user accesses the static HTML page from Amazon S3, which uses the POST method to transfer the form data to API Gateway, which triggers a Lambda function that transforms our data to a format Amazon SageMaker is expecting. Then Amazon SageMaker accepts this input, runs it through our pre-trained model, and produces a new prediction, which is returned to AWS Lambda. AWS Lambda then renders the result to the user.

While this will work well for prototyping, quick demos, and small-scale deployments, we recommend a more robust framework for production environments. To do this, you could forgo the serverless approach, and develop an AWS Elastic Beanstalk frontend for an always-on deployment environment. Alternatively, you could go completely serverless. Going serverless will involve packaging up the contents of your SageMaker endpoint inside of a Lambda function. This process will vary from ML framework to ML framework and is beyond the scope of this blog post. Which route you go will depend on your own production needs.

If you are not intending to leave this application running permanently, don’t forget to clean up the various resources that you have used on Amazon S3 and Amazon SageMaker. Especially make sure that you delete the endpoint so you aren’t changed for its use.

Feel free to use this general structure for building your own, more complex and interesting applications on top of Amazon SageMaker and AWS!
