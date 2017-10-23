# Detroit-Prediction

![alt text](images/detroit.jpg)

## Short summary

The task is to predict whether a given blight ticket will be paid on time. Blight violations are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid.

## Methodology

I started by exploring the dataset. There are a total of 27 columns in the training dataset and 20 columns in the test set. One of the first step will be to remove to columns not present in the test set. Indeed, we cannot train the model with columns absent at time of testing (data leakage).

From a quick look to the missing values, we also notice that the features "violation_zip_code" and "non_us_str_code" have a lot of missing values and will therefore also be removed.

I then merged the train and test data sets with the addresses and latlons datasets. We then had for each ticket id the latitude and longitude for each sample. After a look on Google map, I defined the min and max latitude and longitude for the Detroit city and dropped any samples that stood outside these limits. Finally, I converted the 'lat' and 'lon' columns into bins of hundredth of degrees.

For the features "agency_name", "violation_description", "disposition" and "state", I replaced the values that appear less than five time by a "nan". I then transformed the state into a categorical variable and got dummies values for the "agency_name", "violation_description" and "disposition".

I finally applied a GradientBoostingClassifier with a GridSearch on some hyper-parameters and got a score of 80%.

In the end, I only used six variables for my model and with more domain knowledge, much more feature engineering could be done to obtain a score of at least 80%. For instance, the violation code is certainly an important feature but it has to be cleaned up with care.

## Files details

For all the python files, I used Atom with Hydrogen.

- datasets: datasets used
- exploration.py: python file used for the exploration
- cleaning.py: python file containing the cleaning function to be applied to the training and test set
- randomforest.py: python file that applies the machine learning model to the training set
