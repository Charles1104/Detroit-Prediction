import pandas as pd
import numpy as np
from datetime import datetime

def clean(df_train, df_test, address, latlons):
    # Merge the addresses into the train and test DataFrames. Merge is done with the "ticket_id" column.
    df_train2 = pd.merge(df_train, address, how='inner', left_on='ticket_id', right_on='ticket_id')
    df_test2 = pd.merge(df_test, address, how='inner', left_on='ticket_id', right_on='ticket_id')

    # Merge the latitudes and longitudes into the train and test DataFrames
    df_train3 = pd.merge(df_train2, latlons, how='inner', left_on='address', right_on='address')
    df_test3 = pd.merge(df_test2, latlons, how='inner', left_on='address', right_on='address')

    df_train3 = df_train3.dropna(subset=['lat','lon'])
    df_test3 = df_test3.dropna(subset=['lat','lon'])

    # Remove train data with 'compliance' == NaN
    df_train3 = df_train3.dropna(subset=['compliance'])
    df_train3['compliance'] = df_train3['compliance'].astype(int)

    #create bins for latitude (between 42° and 43°) and longitudes (between -83.8° and -82.7°)
    latbins = np.arange(42, 43.01, 0.01)
    lonbins = np.arange(-83.8, -82.69, 0.01)

    # drop from the training set latitudes and longitudes not belonging to Detroit
    df_train3 = df_train3.drop(df_train3[df_train3.lat < 42].index)
    df_train3 = df_train3.drop(df_train3[df_train3.lat > 43].index)
    df_train3 = df_train3.drop(df_train3[df_train3.lon < -83.8].index)
    df_train3 = df_train3.drop(df_train3[df_train3.lon > -82.7].index)

    # drop from the test set latitudes and longitudes not belonging to Detroit
    df_test3 = df_test3.drop(df_test3[df_test3.lat < 42].index)
    df_test3 = df_test3.drop(df_test3[df_test3.lat > 43].index)
    df_test3 = df_test3.drop(df_test3[df_test3.lon < -83.8].index)
    df_test3 = df_test3.drop(df_test3[df_test3.lon > -82.7].index)

    # apply the binning to the lat and long column of the training set
    df_train3['lat'] = pd.cut(df_train3['lat'], latbins, labels = range(len(latbins)-1))
    df_train3['lon'] = pd.cut(df_train3['lon'], lonbins, labels = range(len(lonbins)-1))

    # apply the binning to the lat and long column of the test set
    df_test3['lat'] = pd.cut(df_test3['lat'], latbins, labels = range(len(latbins)-1))
    df_test3['lon'] = pd.cut(df_test3['lon'], lonbins, labels = range(len(lonbins)-1))

    # make all violation_description codes with < 5 occurrences null in the training set
    counts = df_train3['violation_description'].value_counts()
    df_train3.replace(df_train3['violation_description'][df_train3['violation_description'].isin(counts[counts < 5].index)], np.nan , inplace = True)

    # make all violation_description codes with < 5 occurrences null in the test set
    counts = df_test3['violation_description'].value_counts()
    df_test3.replace(df_test3['violation_description'][df_test3['violation_description'].isin(counts[counts < 5].index)], np.nan , inplace = True)

    # make all agency name codes with < 5 occurrences null in the training set
    counts = df_train3['agency_name'].value_counts()
    df_train3.replace(df_train3['agency_name'][df_train3['agency_name'].isin(counts[counts < 5].index)], np.nan , inplace = True)

    # make all agency name codes with < 5 occurrences null in the test set
    counts = df_test3['agency_name'].value_counts()
    df_test3.replace(df_test3['agency_name'][df_test3['agency_name'].isin(counts[counts < 5].index)], np.nan , inplace = True)

    # make all disposition codes with < 5 occurrences null in the training set
    counts = df_train3['disposition'].value_counts()
    df_train3.replace(df_train3['disposition'][df_train3['disposition'].isin(counts[counts < 5].index)], np.nan , inplace = True)

    # make all disposition codes with < 5 occurrences null in the test set
    counts = df_test3['disposition'].value_counts()
    df_test3.replace(df_test3['disposition'][df_test3['disposition'].isin(counts[counts < 5].index)], np.nan , inplace = True)

    # make all state codes with < 5 occurrences null in the training set
    counts = df_train3['state'].value_counts()
    df_train3.replace(df_train3['state'][df_train3['state'].isin(counts[counts < 5].index)], np.nan , inplace = True)

    df_train3.state.fillna(method = 'pad', inplace = True)

    # make all state codes with < 5 occurrences null in the test set
    counts = df_test3['state'].value_counts()
    df_test3.replace(df_test3['state'][df_test3['state'].isin(counts[counts < 5].index)], np.nan , inplace = True)

    df_test3.state.fillna(method = 'pad', inplace = True)

    # create a new feature equaling the gap in days between the date the ticket has been issued and the hearing date
    def time_gap(input1, input2):
        if not input1 or type(input1)!=str: return 73
        date1 = datetime.strptime(input1, "%Y-%m-%d %H:%M:%S")
        date2 = datetime.strptime(input2, "%Y-%m-%d %H:%M:%S")
        difference = date1 - date2
        return difference.days

    df_train3['time_gap'] = df_train3.apply(lambda row: time_gap(row['hearing_date'], row['ticket_issued_date']), axis = 1).astype(np.float64)
    df_test3['time_gap'] = df_test3.apply(lambda row: time_gap(row['hearing_date'], row['ticket_issued_date']), axis = 1).astype(np.float64)
    # judgbins = np.unique(np.hstack((np.arange(-0.1, 1001.0, 50.1),np.arange(1000.0, 16001.0, 1000.0))))
    # df_train3['judgment_amount'] = pd.cut(df_train3['judgment_amount'], judgbins, labels = range(len(judgbins)-1))
    # df_test3['judgment_amount'] = pd.cut(df_test3['judgment_amount'], judgbins, labels = range(len(judgbins)-1))

    convert_columns={'compliance': 'category',
                    'state': 'category',
                    'zip_code': 'category',
                    'lat': 'category',
                    'lon': 'category'
                    }

    # convert the non-numerical columns to categories
    for df in [df_train3, df_test3]:
        for col, col_type in convert_columns.items():
            if col in df:
                if col_type == 'category':
                    df[col] = df[col].replace(np.nan, "NA", regex=True).astype(col_type)

    # Remove unneeded columns from X sets
    common_cols_to_drop = ['inspector_name', 'mailing_address_str_number',
                           'violator_name', 'violation_street_number', 'violation_street_name', 'violation_code',
                           'mailing_address_str_name', 'address', 'admin_fee', 'violation_zip_code','country',
                           'state_fee', 'late_fee', 'ticket_issued_date', 'hearing_date',
                           'fine_amount', 'clean_up_cost', 'grafitti_status', 'city', 'non_us_str_code']

    y_train = df_train3['compliance']
    train_cols_to_drop = ['payment_status', 'payment_date', 'balance_due', 'payment_amount','compliance', 'compliance_detail', 'collection_status'] + common_cols_to_drop
    df_train3 = df_train3.drop(train_cols_to_drop, axis=1).set_index('ticket_id')
    df_test3 = df_test3.drop(common_cols_to_drop, axis=1).set_index('ticket_id')

    # Convert cetegory columns to integers
    cat_columns = ['state', 'zip_code', 'lat', 'lon']

    #replace the values in the category columns by their cat values
    for df in [df_train3, df_test3]:
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    # List of features for which we will get dummy values
    features_to_be_splitted = ['violation_description', 'agency_name', 'disposition']

    #get dummy values for the training and test data set
    df_train3 = pd.get_dummies(df_train3, columns = features_to_be_splitted, dummy_na = True)
    df_test3 = pd.get_dummies(df_test3, columns = features_to_be_splitted, dummy_na = True)

    train_features = df_train3.columns
    train_features_set = set(train_features)

    # Make sure that both the training and test sest have the same features
    for feature in set(train_features):
        if feature not in df_test3:
            train_features_set.remove(feature)
    train_features = list(train_features_set)

    df_train3 = df_train3[train_features]
    df_test3 = df_test3[train_features]

    return df_train3, df_test3, y_train
