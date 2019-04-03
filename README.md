## Forecasting Weather Using Machine Learning
Forecasting Weather Using Multinomial Logistic Regression, Decision Tree, Naïve Bayes Multinomial, and Support Vector Machine

---
### Data set
Our dataset looks like below which we collected from [Bangladesh Meteorological Department](http://www.bmd.gov.bd/)
![alt text](https://github.com/sksoumik/Forecasting-Weather-Using-Machine-Learning/blob/master/dataset%20sample%20image.PNG)

###### We had last 30 years [1988-2017] of weather data.The training and test set is divided into two segments having 70% and 30% data split across the two categories.
---
### Parameters:
1. Day 
2. Month 
3. Year 
4. Humidity(%) 
5. Max Temp(in ⁰C) 
6. Min Temp(in ⁰C) 
7. Rainfall (in mm) 
8. Sea Level Pressure (in mb)
9. Sunshine (hours) 
 10. Wind Speed(knot) 
 11. Cloud (in okta)
 ---
 ### Train and Test accuracy of the models tested: 

| __Model__ | __Training Accuracy (%)__ | __Testing Accuracy (%)__ |
|-------------|------------|------------|
| Logistic Regression         | 74.2     | 76.9      |
| Decision Tree         | 76.8 | 74.05     |
| Multinomial NB          | 54.27     | 54.34      |
| SVM          | 76.42 | 77.52     |
