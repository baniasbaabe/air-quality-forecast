# Air Quality Forecast

An End-to-End Implementation of Automated Air Quality Forecast for Stuttgart, Germany.

See the web app [here](https://air-quality-forecast.streamlit.app/)

![image](https://github.com/baniasbaabe/air-quality-forecast/assets/72874670/24a6a4eb-9647-4b80-9895-bc5432863dcc)


## Motivation & Goal

There are two main drivers for this project:

1. Stuttgart is well-known for its high density of traffic, and its high level of air pollution in Germany. Multiple fine dust alarms aren't calming. Even if there are ways to combat this problem, it's still worrying.

2. A more technological motivation: I wanted to build an end-to-end Machine Learning project with free tools only. You will see thousands of MLOps (Machine Learning Operations) articles where they will develop projects with tools like AWS, Google Cloud Platform, Airflow, etc... But I also noticed that you will usually have to spend money in some way because the tools (or the deployment of those tools) aren't free (or only free for a limited period). So, the challenge here is to leverage only free tools (or tools with a generous free tier) to build a cool project.

## Data

## Architecture

I used a [3-Pipeline-Architecture](https://www.serverless-ml.org/blog/what-is-serverless-machine-learning), consisting of a Feature Pipeline, Training Pipeline, and Batch Inference Pipeline.

## Limitations

Of course there are a few limitations I want to describe here:

- Since I only utilized free tools, something like a workflow orchestration tool is missing here. GitHub Actions is great for scheduling runs, but it lacks of features you will probably know from Airflow, Prefect, etc.
- For the feature store (Hopsworks) and database (MongoDB) I used, you only have a limited amount of storage for the free tier. So, I clean the whole storage up before a new run starts to not exceed the free memory. Not an optimal solution in a real-world scenario.
- I extract the data from the Feinstaub API. Sometimes, you will get an Timeout if you want to request the data for all Sensor IDs in Stuttgart. Although I tackled this with a retrying mechanism, you will have some moments where it doesn't work.

## Development Set Up
