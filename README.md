# Flight Delays prediction

#### Research question: *What is the influence of the different airlines/airports on the flight delays?*

The goal of this assignment is to to build an interpretable flight prediction model using the [2015 US domestic flight delays and cancellations dataset from Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays/data).

All used data can be found in `Data/`, the two trained MLP models are saved in `model/` and some useful functions used in the notebook are in `helper_functions.py`.

The project itself is divided into 4 notebooks:
1. [Data processing](Data_preprocessing.ipynb) - contains all the required data cleaning and processing pipeline before doing some exploratory analysis and creating the prediction models
2. [Exploratory Data Analysis](Exploratory_Data_Analysis.ipynb)- allows to explore our flights dataset and investigate some inital relations between airlines/airports and flight delays
3. [Baseline prediction model](PredictionModel_baseline.ipynb) - sets up the baseline flight delay prediction model using a Decision Tree and provide first insights on feature importance
4. [Neural Network prediction model](PredictionModel_complex.ipynb) - contains the implementation of an interpretable MLP to predict the fligh delays as well as get some insights about its predictions
