
# NYC Traffic Collision Analysis

## Overview

Urban traffic collisions pose significant safety concerns. This project utilizes the extensive NYPD collision dataset to conduct a comprehensive analysis up to October 9th, 2023. The analysis focuses on:

Identifying Common Vehicle Types in Collisions: Analyzing the types of vehicles most commonly involved in collisions.
Developing Predictive Models for High-risk Area Identification: Creating predictive models to identify high-risk areas for collisions.  
The insights from this analysis aim to facilitate more informed conclusions and recommendations for improving road safety in New York City.
## Tools and Technologies
Language: Python        
Tools: Jupyter Notebook     
Framework: Flask


## dataset

The dataset used for this analysis can be found on the NYC Open Data portal.    
https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data

## Approach

Understanding the Data - It is important to understand our data and our problem statement i.e., What are the most common types of vehicles involved in collisions in New York City?

Preparing the Data - After understanding our dataset, it is essential to prepare the data. We have used EDA techniques to remove null values and duplicate entries and explore relationships between variables and assess their distributions across different categories.

Perform Analysis - We have carried out a Time-series analysis, univariate and Bivariate analysis to understand more about the factors and causes of Motor collisions in New York City. We evaluated the data using various ML algorithms.

Give Recommendations - Based on our analysis, we will provide insights and recommendations to decrease the number of Motor collisions.

## workflow

Data Collection     
We gathered information about accidents from the NYC Open Data database and collected weather details from Weather Underground. This allowed us to analyze whether weather conditions during accidents influenced their severity.

Data Balancing          
The initial dataset was imbalanced, with more instances of one type of accident than the other. To address this, we balanced the dataset to ensure an equal representation of both accident types.

Data Analysis and Modeling      
Using the balanced data, we trained machine learning models to predict accident severity based on weather conditions and time of day. Our models achieved an accuracy of approximately 72%.

Classifier Development      
Upon completing the modeling process, we developed a classifier capable of identifying high-risk areas for severe collisions. This classifier can aid authorities in implementing precautionary measures, such as speed limits and vehicle size restrictions, to reduce the likelihood of accidents.


<img width="1399" alt="Flask web page result" src="https://github.com/PoojithLankipalli/NYC-Traffic-Collision-Analysis/assets/69042617/6b2aeebb-8b43-488d-a324-c10497514dd2">









