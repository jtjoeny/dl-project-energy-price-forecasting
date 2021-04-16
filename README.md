# Reproducibility project: Deep Learning-Based Multivariate Probabilistic Forecasting for Short-Term Scheduling in Power Markets

*Course: 				Deep Learning
Course code:		CS4240
Name: 				Jamie Tjoe Ny
Student ID: 		4400356*

Reproduction of existing papers, especially those without code published, forms an important part of Deep Learning research. This blogpost will discuss the reproducibility of the paper "[Deep Learning-Based Multivariate Probabilistic Forecasting for Short-Term Scheduling in Power Markets](https://www.researchgate.net/publication/327635519_Deep_Learning-Based_Multivariate_Probabilistic_Forecasting_for_Short-Term_Scheduling_in_Power_Markets)"

The goal of this project is to achieve a similar result as the paper shows in figure 6d
![Figure 6d](https://pasteboard.co/JXrHL3l.png)

This graph shows a plot of quantiles for the day ahead price forecast and the actual price as a line.

In this blogpost I will explain how I approached the paper, project and the decisions I made.

## Challenges regarding the paper

Before you can begin with the reproducibility project of a paper, you first need to understand the contents of the paper. I approached this by reading the paper part by part, spread out over multiple days, as this paper is quite dense and highly technical. 

Prior to the part of the model that produces the quantile plot of the day-ahead prices, more probabilistic modeling is performed in the paper. However, I found it not very clear which exact steps the authors have taken. Luckily, in consultation with the external supervisor and the other groups working on this project, it was decided that we should only focus on producing the quantile plot using weather data and price data, using a bi-LSTM network. This meant I didn't have to impelment the copula algorithm, which the paper vaguely describes. Although I was a bit concerned dropping this step would possibly decrease the performance and quality of results of this project, it did make the project easier to understand. This new information gave me a more clear path to follow!

## Gathering of data
For this project, a combination of two independent data sources is being used. Firstly, historical weather data (09-01-2011 - 31-12-2016) from WorldWeatherOnline is being used, from a weather station located in Brussels. The weather data is provided with an hourly interval. I picked Brussels because of it's central location, as the energy price data will be nation wide. Another idea could be to take data from multiple weather stations, spread out over the country, and take the average of these. 

Next, the historical energy prices are an important data source for this project. These prices are from the EPEX-Belgium energy market, covering the same period as the weather data.  The energy prices are hourly. 

## Weather data from WorldWeatherOnline
The weather data is taken from WorldWeatherOnline, through their API interface. The data is gathered from a Brussels weather station and includes temperature, cloud cover, wind speed, precipation and more. Including all the weather data seems a bit extensive, therefore, I made a selection of some features to include for this project. The selected weather features are 'sunHour', 'cloudcover', 'humidity', 'tempC' and 'windspeedKmph'. This means that some features are dropped, such as WindChill, FeelsLike temperature and more. If you want to contine on my work, it could be a good idea to try more variations of weather features as input, possibly reducing it to only 2 or 3 or increasing it to see its effect. 

## Day-ahead price from ENTSOE
The day-ahead price data is taken from the European Network of Transmission System Operators for Electricity (ENTSOE). The ENTSOE platform ensures a transparent energy market which contributes to competitivety and therefore ensures fair prices. For the Belgian market, or EPEX-BE, I use the historical day-ahead energy prices for the period of 09-01-2011 until 31-12-2016. In total, this consists of 52416 data points.

The day-ahead price predictions are used by network operators and allows them to take a position in the market. Based on the expected offer and demand, the day-ahead energy market lets market participants commit to buy or sell elecrticity one day before the operating day. This helps to avoid price volatility and lets the market produce one financial settlement. The more accurate the price prediction of a market participant is, the more profit the participant can make in potential.
The dataset I used was taken from the [epftoolbox] (https://github.com/jeslago/epftoolbox) project as it already did one preprocessing step: "the daylight saving times (DST) are pre-processed by interpolating the missing values in Spring and averaging the values corresponding to the duplicated time indices in Autumn".



>>Plot weather data
> Plot energy prices

## Data preparation
As the data is taken from two different sources, some cleaning and preparation had to be done. First I made sure both datasets (weather and energy prices) had timesteps with the same interval (hourly). Next, the columns with unused weather features were dropped in the weather dataset. The energy price dataset consists of data from the same period as the weather dataset and have daylight saving times pre-processed as described above. The datasets were checked to be from the same timezone and were merged based on their timestamp. Before training the model, all features were normalized with a MinMaxScaler.
Now we have one dataset of size (52416,6), namely the 5 weather features and the historical energy prices. The datetime is set as index for the dataframe. In order to prepare the data for training, validation and testing, the dataset is split in this way respectively.  85% of the data is reserved for training and validation where 15% of the data is reserved for testing the performance of the model. From the training and validation set, a training-validation split of 0.8 is used. Overall, the train-val-test ratio boils down to 68% - 17% - 15%. This ratio is based on research and similar projects and papers. 

The train, validation and tests sets are modified such that each block of data consists of 24 hours of previous data, after which a prediction will be made 24 hours forward. Given the past 24 hours, the model therefore will be able to produce 24 hours forward. This way, the model can always make a day-ahead prediction.

## Model creation
Now we have the dataset ready, we will now discuss how the model is set up and what its architecture looks like.  The model consists of a bi-LSTM layer, a dense layer, an activation function and a loss function.

###  Bi-LSTM layer
The bidirectional-LSTM layer is a variation on the normal LSTM layer, which consists of long-short-term-memory cells with gates. A bidrectional LSTM processes data in both directions, left to right and right to left, so to say. A normal LSTM processes data unidirectional. A LSTM is suited well for analyzing time series data due to its ability to find pattern over long periods of time.

### Dense layer
A dense layer is a layer of fully connected neurons. In a dense layer, inputs are multiplied with weights and the respective biases are added, after which the outcome goes through an activation function.

### Activation function
The activation function will produce the final output of the dense layer. In my experiments, I chose a ReLU activation layer. 

### Loss function
For the price prediction part of the model, the 'mean aboslute error' loss function is being used, as implemented by Keras. Hower, in order to predict the quantiles, a different loss function is required. Therefore, I adapted my quantile loss function from this [well explained IPYNB notebook](https://github.com/sachinruk/KerasQuantileModel/blob/master/Keras%20Quantile%20Model.ipynb). 

# Hyperparameter tuning
In the first stage of this project, I performed a lot of manual experimentation in order to find the best model. Sometimes I had success, but it did not feel like a proper approach in finding good hyperparameters. In consultation with my TA, I decided to perform another step and implement hyperparameter tuning. As I already built previous parts of the project with Keras, I searched for a tuner package within Keras. 

In order to find the optimal amount of neurons for each layer, the best learning rate and the optimal amount of epochs to fit the model with, I did some hyperparameter tuning. In order to do this, I followed the [Introduction to Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) and decided to use the Hyperband tuner. I tuned for the optimal amount of neurons in the bi-LSTM layer, which turned out to be 320. The optimal learning rate for the optimizer turned out to be 0.001. 

The optimal amount of epochs to fit the model with was searched for two times. First, I found 32 as the optimal number of epochs, the second time 12 turned out to be the optimal number of epochs. After evaluation both models (fitted with 12 and 32 epochs) on the test dataset, I decided to go with the model fitted 32 epochs. 

# Final experiments
Now I have explained how I prepared the data, set up the model and how I selected the hyperparameters, it is time to talk about the results produced by the model.  The model is able to predict a price 24 hours ahead or to predict quantiles 24 hours ahead. 

The quantiles plot including the actual price is produced for the first 7 days of the test data period (08-02-2016 10h00 until 15-02-2016 10h00). Note that the graph included shows the quantiles and actual price for the normalized data.
