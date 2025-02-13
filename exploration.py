import pandas as pd
import numpy as np
from datetime import datetime

train_data = pd.read_csv('datasets/train.csv', encoding = 'ISO-8859-1')
print('shape', train_data.shape)
train_data.info()

# We can see that the dataset contains 250306 samples in total but it has only 159880 samples for
# the target value. We will have to remove all the samples with a null value during cleaning.
# We can also notice that the feature "violaton_zip_code" and "non_us_str_code" have a lot of missing
# values and will therefore also be removed.

train_data.columns

test_data = pd.read_csv('datasets/test.csv')
test_data.info()

#colums not in common
col_diff = [x for x in train_data.columns if x not in test_data.columns]
col_diff

# We will have to remove all the columns that are in the train data but not in the test data.
# Otherwise this will be considered as data leakage as th emodel might learn from data that is
# not available at time of the prediction.

#columns in common that have to be used by the machine learning
col_comm = [x for x in train_data.columns if x in test_data.columns]
col_comm

address =  pd.read_csv('datasets/addresses.csv')
address.head()

latlons = pd.read_csv('datasets/latlons.csv')
latlons.head()

# Merge the addresses into the train and test DataFrames. Merge is done with the "ticket_id" column.
train_data = pd.merge(train_data, address, how='inner', left_on='ticket_id', right_on='ticket_id')
test_data = pd.merge(test_data, address, how='inner', left_on='ticket_id', right_on='ticket_id')

# Merge the latitudes and longitudes into the train and test DataFrames
train_data = pd.merge(train_data, latlons, how='inner', left_on='address', right_on='address')
test_data = pd.merge(test_data, latlons, how='inner', left_on='address', right_on='address')

train_data['lat'].value_counts()

sum(train_data['lat'].isnull())
sum(train_data['lon'].isnull())

# Samples with missing values for lat and lon have to be removed
# bins have to be created for the latitudes and longitudes. One bin per hundreth of latitude and longitude
# Max latitude coordinate should be 43° and min latitude 42° (100 bins)
# Min longitude coordinate should be -83.8° and max latitude -82.7 (110 bins)

counts = train_data['violation_description'].value_counts()

train_data['violation_description'].isin(counts[counts < 5].index)

a = datetime.strptime(train_data['hearing_date'][0], "%Y-%m-%d %H:%M:%S")
b = datetime.strptime(train_data['ticket_issued_date'][0], "%Y-%m-%d %H:%M:%S")
a
b
(a - b).days
