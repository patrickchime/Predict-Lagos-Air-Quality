# Predicting Air Quality in Lagos, Nigeria.

## Introduction
Particulate Matter 2.5 (PM2.5) refers to tiny particles or droplets in the air that are two and a half microns or less in width. These particles can pose health risks when inhaled, especially when they are at elevated levels. Understanding the temporal patterns of PM2.5 concentration can provide valuable insights into its sources and potential impacts on human health.

Accurate monitoring of PM2.5 (Particulate Matter 2.5 micrometers or smaller in diameter) is indeed a crucial step towards pollution control, climate change control and overall environmental quality improvement in Lagos,Nigeria.

Monitoring PM2.5 aligns with international commitments to combat climate change and improve air quality. Lagos, like many regions, may be part of international agreements and initiatives that require reporting on air quality and emissions reduction efforts.


## About Dataset
This dataset is sourced from [openAfrica](https://open.africa/dataset/sensorsafrica-airquality-archive-lagos). 
This data set contains PM (particulate matter), temperature, and humidity readings taken with low-cost sensors. These sensors measure the concentration of PM in the air, including particles with diameters less than or equal to 1 micrometer (PM1), 2.5 micrometers (PM2.5), and particles with diameters less than or equal to 10 micrometers (PM10). The data set includes information on the sensor type, date, time, and location of the readings, as well as the sensor’s specific measurement values for Temperature (C), Humidity (%), PM1, PM2.5, and PM10.

## Objective

The objective of this work is to develop a predictive model using dataset of emissions,  to forecast pm2.5 in Lagos. A simple User Interface will be develop for easy navigation.

This work can enable authorities to respond swiftly to pollution events, enforce regulations, and allocate resources efficiently for pollution control and climate change adaptation measures.

## Conclusion
## Conclusion

1. While the **`WHO`** guideline for the annual mean PM 2.5 concentration level is 5 µg/m3 and 24-hour average exposures should not exceed 15µg/m3 more than 3-4 days per year, the average hourly PM 2.5 concentration in Lagos has been above the WHO baseline every day up to the last day in the dataset.

2. The data is modelled using SARIMAX since there is a presence of weekly trend in the dataset.
3. The model was deployed to the cloud using Streamlit for easy navigation and prediction by end users.
