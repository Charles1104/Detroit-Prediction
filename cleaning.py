import pandas as pd
import numpy as np

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

    df_train3 = df_train3.drop(df_train3[df_train3.lat < 42].index)
    df_train3 = df_train3.drop(df_train3[df_train3.lat > 43].index)
    df_train3 = df_train3.drop(df_train3[df_train3.lon < -83.8].index)
    df_train3 = df_train3.drop(df_train3[df_train3.lon > -82.7].index)

    df_test3 = df_test3.drop(df_test3[df_test3.lat < 42].index)
    df_test3 = df_test3.drop(df_test3[df_test3.lat > 43].index)
    df_test3 = df_test3.drop(df_test3[df_test3.lon < -83.8].index)
    df_test3 = df_test3.drop(df_test3[df_test3.lon > -82.7].index)

    df_train3['lat'] = pd.cut(df_train3['lat'], latbins, labels = range(len(latbins)-1))
    df_train3['lon'] = pd.cut(df_train3['lon'], lonbins, labels = range(len(lonbins)-1))

    df_test3['lat'] = pd.cut(df_test3['lat'], latbins, labels = range(len(latbins)-1))
    df_test3['lon'] = pd.cut(df_test3['lon'], lonbins, labels = range(len(lonbins)-1))

    judgbins = np.unique(np.hstack((np.arange(-0.1, 1001.0, 50.1),np.arange(1000.0, 16001.0, 1000.0))))
    df_train3['judgment_amount'] = pd.cut(df_train3['judgment_amount'], judgbins, labels = range(len(judgbins)-1))
    df_test3['judgment_amount'] = pd.cut(df_test3['judgment_amount'], judgbins, labels = range(len(judgbins)-1))

    convert_columns={'compliance': 'category',
                    'state': 'category',
                    'zip_code': 'category',
                    }

    for df in [df_train3, df_test3]:
        for col, col_type in convert_columns.items():
            if col in df:
                if col_type == 'category':
                    df[col] = df[col].replace(np.nan, "NA", regex=True).astype(col_type)

    # Remove unneeded columns from X sets
    common_cols_to_drop = ['agency_name', 'inspector_name', 'mailing_address_str_number',
                           'violator_name', 'violation_street_number', 'violation_street_name', 'violation_code',
                           'mailing_address_str_name', 'address', 'admin_fee', 'violation_zip_code','country',
                           'state_fee', 'late_fee', 'ticket_issued_date', 'hearing_date', 'violation_description',
                           'fine_amount', 'clean_up_cost', 'disposition', 'grafitti_status', 'city', 'non_us_str_code']

    y_train = df_train3['compliance']
    train_cols_to_drop = ['payment_status', 'payment_date', 'balance_due', 'payment_amount','compliance', 'compliance_detail', 'collection_status'] + common_cols_to_drop
    df_train3 = df_train3.drop(train_cols_to_drop, axis=1).set_index('ticket_id')
    df_test3 = df_test3.drop(common_cols_to_drop, axis=1).set_index('ticket_id')

    # Convert cetegory columns to integers
    cat_columns = ['state', 'zip_code']

    for df in [df_train3, df_test3]:
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    return df_train3, df_test3, y_train
