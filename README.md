# Enefit - Predict Energy Behavior of Prosumers Solution

### Competition Overview

**The goal of this competition is to create a model that predicts the energy usage of prosumers (energy consumers who also produce energy), in order to reduce the cost of energy imbalance.**

This competition aims to address the problem of energy imbalance, which refers to the inconsistency between the expected and actual usage or production of energy. Prosumers, who both consume and generate energy, are a major cause of energy imbalance. Although they make up a small portion of all consumers, their unpredictable behavior creates logistical and financial challenges for energy companies.

### Evaluation Metric

The evaluation is based on the Mean Absolute Error (MAE) between the predicted returns and observed targets:

$$
MAE = \frac{1}{n} \sum\limits_{i=1}^{n} {|y_i - x_i|}
$$

where:

- \( n \) is the total number of data points.
- \( y_i \) is the predicted value for the \( i \)-th data point.
- \( x_i \) is the observed value for the \( i \)-th data point.

**Submission**  
You must use the provided Python time series API to submit to this competition, to ensure the model doesn’t peek into the future. Follow the template in the provided notebook to use the API.

### Data Description

In this competition, your task is to predict the electricity consumption and production of Estonian customers with installed solar panels. You have access to weather data, related energy prices, and records of installed PV capacity.

This is a forecasting competition using a time series API. The private leaderboard will use actual data collected after the submission period ends.

💡 Note: All datasets follow the same time convention. Timestamps are given in EET/EEST (Eastern European Time / Eastern European Summer Time). Most variables are hourly sums or averages. All datetime columns indicate the **start** of a one-hour period, **except for weather data**, where some variables (e.g., temperature, cloud cover) represent the **end** of the period.

...

### Summary

In the Enefit energy behavior prediction competition, we built a comprehensive time series prediction pipeline to forecast energy consumption and production. The project combined extensive data cleaning and feature engineering from multiple data sources including customer info, weather forecasts, historical weather, energy prices, and holidays.

Efficient data processing was handled with `polars`, and we constructed a rich feature set including time features, client attributes, weather conditions, and lagged targets. The modeling phase used `LightGBM`, incorporating various lag-difference features to enhance prediction accuracy. Final predictions were blended and clipped to ensure stability. This project improved understanding of time series prediction, feature engineering, and model tuning using LightGBM, significantly boosting forecast accuracy and reducing energy imbalance costs.
