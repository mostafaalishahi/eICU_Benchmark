from __future__ import absolute_import
from __future__ import print_function


import os
import pandas as pd
import numpy as np
import sys
import shutil

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals.joblib import dump, load


def dataframe_from_csv(path, header=0, index_col=False):
    return pd.read_csv(path, header=header, index_col=index_col)


var_to_consider = ['glucose', 'Invasive BP Diastolic', 'Invasive BP Systolic',
                   'O2 Saturation', 'Respiratory Rate', 'Motor', 'Eyes', 'MAP (mmHg)',
                   'Heart Rate', 'GCS Total', 'Verbal', 'pH', 'FiO2', 'Temperature (C)']



#Filter on useful column for this benchmark
def filter_patients_on_columns_model(patients):
    columns = ['patientunitstayid', 'gender', 'age', 'ethnicity', 'apacheadmissiondx',
               'admissionheight', 'hospitaladmitoffset', 'admissionweight',
               'hospitaldischargestatus', 'unitdischargeoffset', 'unitdischargestatus']
    return patients[columns]

#Select unique patient id
def cohort_stay_id(patients):
    cohort = patients.patientunitstayid.unique()
    return cohort

#Convert gender from F/M to numbers
g_map = {'Female': 1, 'Male': 2, '': 0, 'NaN': 0, 'Unknown': 0, 'Other': 0}
def transform_gender(gender_series):
    global g_map
    return {'gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map[''])}


#Convert ethnicity to numbers
e_map = {'Asian': 1, 'African American': 2, 'Caucasian': 3, 'Hispanic': 4, 'Native American': 5, 'NaN': 0, '': 0}
def transform_ethnicity(ethnicity_series):
    global e_map
    return {'ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map[''])}


#Convert hospital/unit discharge status into numbers
h_s_map = {'Expired': 1, 'Alive': 0, '': 2, 'NaN': 2}
def transform_hospital_discharge_status(status_series):
    global h_s_map
    return {'hospitaldischargestatus': status_series.fillna('').apply(
        lambda s: h_s_map[s] if s in h_s_map else h_s_map[''])}

def transform_unit_discharge_status(status_series):
    global h_s_map
    return {'unitdischargestatus': status_series.fillna('').apply(
        lambda s: h_s_map[s] if s in h_s_map else h_s_map[''])}


# Convert diagnosis into numbers
def transform_dx_into_id(df):
    df.apacheadmissiondx.fillna('nodx', inplace=True)
    dx_type = df.apacheadmissiondx.unique()
    dict_dx_key = pd.factorize(dx_type)[1]
    dict_dx_val = pd.factorize(dx_type)[0]
    dictionary = dict(zip(dict_dx_key, dict_dx_val))
    df['apacheadmissiondx'] = df['apacheadmissiondx'].map(dictionary)
    return df


## Extract the root data

#Extract data from patient table
def read_patients_table(eicu_path, output_path):
    pats = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv'), index_col=False)
    pats = filter_patients_on_age(pats, min_age=18, max_age=89)
    pats = filter_one_unit_stay(pats)
    pats = filter_patients_on_columns(pats)
    pats.update(transform_gender(pats.gender))
    pats.update(transform_ethnicity(pats.ethnicity))
    pats.update(transform_hospital_discharge_status(pats.hospitaldischargestatus))
    pats.update(transform_unit_discharge_status(pats.unitdischargestatus))
    pats = transform_dx_into_id(pats)
    pats.to_csv(os.path.join(output_path, 'all_stays.csv'), index=False)
    pats = filter_patients_on_columns_model(pats)
    return pats

#Select unique patient id
def cohort_stay_id(patients):
    cohort = patients.patientunitstayid.unique()
    return cohort

#filter on adult patients
def filter_patients_on_age(patient, min_age=18, max_age=89):
    patient.ix[patient['age'] == '> 89', 'age'] = 90
    patient[['age']] = patient[['age']].fillna(-1)
    patient[['age']] = patient[['age']].astype(int)
    patient = patient.ix[(patient.age >= min_age) & (patient.age <= max_age)]
    return patient

#filter those having just one stay in unit
def filter_one_unit_stay(patients):
    cohort_count = patients.groupby(by='uniquepid').count()
    index_cohort = cohort_count[cohort_count['patientunitstayid'] == 1].index
    patients = patients[patients['uniquepid'].isin(index_cohort)]
    return patients

#Filter on useful columns from patient table
def filter_patients_on_columns(patients):
    columns = ['patientunitstayid', 'gender', 'age', 'ethnicity', 'apacheadmissiondx',
               'hospitaladmityear', 'hospitaldischargeyear', 'hospitaldischargeoffset',

               'admissionheight', 'hospitaladmitoffset', 'admissionweight',
               'hospitaldischargestatus', 'unitdischargeoffset', 'unitdischargestatus']
    return patients[columns]

#Write the selected cohort data from patient table into pat.csv for each patient
def break_up_stays_by_unit_stay(pats, output_path, stayid=None, verbose=1):
    unit_stays = pats.patientunitstayid.unique() if stayid is None else stayid
    nb_unit_stays = unit_stays.shape[0]
    for i, stay_id in enumerate(unit_stays):
        if verbose:
            sys.stdout.write('\rStayID {0} of {1}...'.format(i + 1, nb_unit_stays))
        dn = os.path.join(output_path, str(stay_id))
        try:
            os.makedirs(dn)
        except:
            pass

        pats.ix[pats.patientunitstayid == stay_id].sort_values(by='hospitaladmitoffset').to_csv(
            os.path.join(dn, 'pats.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')


## Here we deal with lab table
#Select the useful columns from lab table
def filter_lab_on_columns(lab):
    columns = ['patientunitstayid', 'labresultoffset', 'labname', 'labresult']
    return lab[columns]

#Rename the columns in order to have a unified name
def rename_lab_columns(lab):
    lab.rename(index=str, columns={"labresultoffset": "itemoffset",
                                   "labname": "itemname", "labresult": "itemvalue"}, inplace=True)
    return lab

#Select the lab measurement from lab table
def item_name_selected_from_lab(lab, items):
    lab = lab[lab['itemname'].isin(items)]
    return lab

#Check if the lab measurement is valid
def check(x):
    try:
        x = float(str(x).strip())
    except:
        x = np.nan
    return x
def check_itemvalue(df):
    df['itemvalue'] = df['itemvalue'].apply(lambda x: check(x))
    df['itemvalue'] = df['itemvalue'].astype(float)
    return df

#extract the lab items for each patient
def read_lab_table(eicu_path):
    lab = dataframe_from_csv(os.path.join(eicu_path, 'lab.csv'), index_col=False)
    items = ['bedside glucose', 'glucose', 'pH', 'FiO2']
    lab = filter_lab_on_columns(lab)  
    lab = rename_lab_columns(lab)
    lab = item_name_selected_from_lab(lab, items)  
    lab.loc[lab['itemname'] == 'bedside glucose', 'itemname'] = 'glucose'  # unify bedside glucose and glucose
    lab = check_itemvalue(lab)
    return lab


#Write the available lab items of a patient into lab.csv 
def break_up_lab_by_unit_stay(lab, output_path, stayid=None, verbose=1):
    unit_stays = lab.patientunitstayid.unique() if stayid is None else stayid
    nb_unit_stays = unit_stays.shape[0]
    for i, stay_id in enumerate(unit_stays):
        if verbose:
            sys.stdout.write('\rStayID {0} of {1}...'.format(i + 1, nb_unit_stays))
        dn = os.path.join(output_path, str(stay_id))
        try:
            os.makedirs(dn)
        except:
            pass
        lab.ix[lab.patientunitstayid == stay_id].sort_values(by='itemoffset').to_csv(os.path.join(dn, 'lab.csv'),
                                                                                     index=False)
    if verbose:
        sys.stdout.write('DONE!\n')



#Filter the useful columns from nc table
def filter_nc_on_columns(nc):
    columns = ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevallabel',
               'nursingchartcelltypevalname', 'nursingchartvalue']
    return nc[columns]

#Unify the column names in order to be used later
def rename_nc_columns(nc):
    nc.rename(index=str, columns={"nursingchartoffset": "itemoffset",
                                  "nursingchartcelltypevalname": "itemname",
                                  "nursingchartcelltypevallabel": "itemlabel",
                                  "nursingchartvalue": "itemvalue"}, inplace=True)
    return nc

#Select the items using name and label
def item_name_selected_from_nc(nc, label, name):
    nc = nc[(nc.itemname.isin(name)) |
            (nc.itemlabel.isin(label))]
    return nc

#Convert fahrenheit to celsius
def conv_far_cel(nc):
    nc['itemvalue'] = nc['itemvalue'].astype(float)
    nc.loc[nc['itemname'] == "Temperature (F)", "itemvalue"] = ((nc['itemvalue'] - 32) * (5 / 9))

    return nc

#Unify the different names into one for each measurement
def replace_itemname_value(nc):  
    nc.loc[nc['itemname'] == 'Value', 'itemname'] = nc.itemlabel
    nc.loc[nc['itemname'] == 'Non-Invasive BP Systolic', 'itemname'] = 'Invasive BP Systolic'
    nc.loc[nc['itemname'] == 'Non-Invasive BP Diastolic', 'itemname'] = 'Invasive BP Diastolic'
    nc.loc[nc['itemname'] == 'Temperature (F)', 'itemname'] = 'Temperature (C)'
    nc.loc[nc['itemlabel'] == 'Arterial Line MAP (mmHg)', 'itemname'] = 'MAP (mmHg)'
    return nc


#Select the nurseCharting items and save it into nc
def read_nc_table(eicu_path):
    # import pdb;pdb.set_trace()
    nc = dataframe_from_csv(os.path.join(eicu_path, 'nurseCharting.csv'), index_col=False)
    nc = filter_nc_on_columns(nc)
    nc = rename_nc_columns(nc)
    typevallabel = ['Glasgow coma score', 'Heart Rate', 'O2 Saturation', 'Respiratory Rate', 'MAP (mmHg)',
                    'Arterial Line MAP (mmHg)']
    typevalname = ['Non-Invasive BP Systolic', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic',
                   'Invasive BP Diastolic', 'Temperature (C)', 'Temperature (F)']
    nc = item_name_selected_from_nc(nc, typevallabel, typevalname)
    nc = check_itemvalue(nc) 
    nc = conv_far_cel(nc)
    replace_itemname_value(nc)
    del nc['itemlabel']
    return nc


#Write the nc values of each patient into a nc.csv file
def break_up_stays_by_unit_stay_nc(nursecharting, output_path, stayid=None, verbose=1):
    unit_stays = nursecharting.patientunitstayid.unique() if stayid is None else stayid
    nb_unit_stays = unit_stays.shape[0]
    for i, stay_id in enumerate(unit_stays):
        if verbose:
            sys.stdout.write('\rStayID {0} of {1}...'.format(i + 1, nb_unit_stays))
        dn = os.path.join(output_path, str(stay_id))
        try:
            os.makedirs(dn)
        except:
            pass

        nursecharting.ix[nursecharting.patientunitstayid == stay_id].sort_values(by='itemoffset').to_csv(
            os.path.join(dn, 'nc.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')


# Write the time-series data into one csv for each patient
def extract_time_series_from_subject(t_path):
    print("Convert to time series ...")
    print("This will take some hours, as the imputation and binning and converting time series are done here ...")

    filter_15_200 = 0

    for i,stay_dir in enumerate(os.listdir(t_path)):
        # import pdb;pdb.set_trace()
        dn = os.path.join(t_path, stay_dir)
        try:
            stay_id = int(stay_dir)
            if not os.path.isdir(dn):
                raise Exception
        except:
            continue
        try:
            pat = dataframe_from_csv(os.path.join(t_path, stay_dir, 'pats.csv'))
            lab = dataframe_from_csv(os.path.join(t_path, stay_dir, 'lab.csv'))
            nc = dataframe_from_csv(os.path.join(t_path, stay_dir, 'nc.csv'))
            nclab = pd.concat([nc, lab]).sort_values(by=['itemoffset'])
            timeepisode = convert_events_to_timeseries(nclab, variables=var_to_consider)
            nclabpat = pd.merge(timeepisode, pat, on='patientunitstayid')
            df = binning(nclabpat, 60)
            df = imputer(df, strategy='normal')
            if 15 <= df.shape[0] <= 200:
                filter_15_200 += 1
                df = check_in_range(df)
                df.to_csv(os.path.join(t_path, stay_dir, 'timeseries.csv'), index=False)
                sys.stdout.write('\rWrite patient {0} / {1}'.format(i,len(os.listdir(t_path))))
            else:
                continue
        except:
            continue
    print("Number of patients with less than 15 or more than 200 records:",filter_15_200)
    print('Convereted to time series')


##Convert to time-series

#Check the range of each measurment
def check_in_range(df):
    df['Eyes'].clip(0, 5, inplace=True)
    df['GCS Total'].clip(2, 16, inplace=True)
    df['Heart Rate'].clip(0, 350, inplace=True)
    df['Motor'].clip(0, 6, inplace=True)
    df['Invasive BP Diastolic'].clip(0, 375, inplace=True)
    df['Invasive BP Systolic'].clip(0, 375, inplace=True)
    df['MAP (mmHg)'].clip(14, 330, inplace=True)
    df['Verbal'].clip(1, 5, inplace=True)
    df['admissionheight'].clip(100, 240, inplace=True)
    df['admissionweight'].clip(30, 250, inplace=True)
    df['glucose'].clip(33, 1200, inplace=True)
    df['pH'].clip(6.3, 10, inplace=True)
    df['FiO2'].clip(15, 110, inplace=True)
    df['O2 Saturation'].clip(0, 100, inplace=True)
    df['Respiratory Rate'].clip(0, 100, inplace=True)
    df['Temperature (C)'].clip(26, 45, inplace=True)
    return df

#Read each patient nc, lab and demographics and put all in one csv
def convert_events_to_timeseries(events, variable_column='itemname', variables=[]):
    metadata = events[['itemoffset', 'patientunitstayid']].sort_values(
        by=['itemoffset', 'patientunitstayid']).drop_duplicates(keep='first').set_index('itemoffset')
        
    timeseries = events[['itemoffset', variable_column, 'itemvalue']].sort_values(
        by=['itemoffset', variable_column, 'itemvalue'], axis=0).drop_duplicates(subset=['itemoffset', variable_column],
                                                                                 keep='last')
    timeseries = timeseries.pivot(index='itemoffset', columns=variable_column, values='itemvalue').merge(metadata,
                                                                                                         left_index=True,
                                                                                                         right_index=True).sort_index(
        axis=0).reset_index()
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries

#Bin all the values of one hour, into one bin
def binning(df, x=60):
    null_columns = ['glucose', 'Invasive BP Diastolic', 'Invasive BP Systolic',
                    'O2 Saturation', 'Respiratory Rate', 'Motor', 'Eyes', 'MAP (mmHg)',
                    'Heart Rate', 'GCS Total', 'Verbal', 'pH', 'FiO2', 'Temperature (C)']

    df['glucose'] = df['glucose'].shift(-1)
    df.dropna(how='all', subset=null_columns, inplace=True)
    df['itemoffset'] = (df['itemoffset'] / x).astype(int)
    df = df.groupby('itemoffset').apply(lambda x: x.fillna(x.mean()))
    df = df.droplevel(0, axis=0)
    df.drop_duplicates(subset=['itemoffset'], keep='last', inplace=True)
    return df

#Imputation
def imputer(dataframe, strategy='zero'):
    normal_values = {'Eyes': 4, 'GCS Total': 15, 'Heart Rate': 86, 'Motor': 6, 'Invasive BP Diastolic': 56,
                     'Invasive BP Systolic': 118, 'O2 Saturation': 98, 'Respiratory Rate': 19,
                     'Verbal': 5, 'glucose': 128, 'admissionweight': 81, 'Temperature (C)': 36,
                     'admissionheight': 170, "MAP (mmHg)": 77, "pH": 7.4, "FiO2": 0.21}

    if strategy not in ['zero', 'back', 'forward', 'normal']:
        raise ValueError("impute strategy is invalid")
    df = dataframe
    if strategy in ['zero', 'back', 'forward', 'normal']:
        if strategy == 'zero':
            df.fillna(value=0, inplace=True)
        elif strategy == 'back':
            df.fillna(method='bfill', inplace=True)
        elif strategy == 'forward':
            df.fillna(method='ffill', inplace=True)
        elif strategy == 'normal':
            df.fillna(value=normal_values, inplace=True)
        if df.isna().sum().any():
            df.fillna(value=normal_values, inplace=True)
        return df


#Delete folders without timeseries file
def delete_wo_timeseries(t_path):
    # import pdb;pdb.set_trace()
    for stay_dir in os.listdir(t_path):
        dn = os.path.join(t_path, stay_dir)
        try:
            stay_id = int(stay_dir)
            if not os.path.isdir(dn):
                raise Exception
        except:
            continue
        try:
            sys.stdout.flush()
            if not os.path.isfile(os.path.join(dn, 'timeseries.csv')):
                shutil.rmtree(dn)
        except:
            continue
    print('DONE deleting')

#Write all the extracted data into one csv file
def all_df_into_one_df(output_path):
    all_filenames = []
    unit_stays = pd.Series(os.listdir(output_path))
    unit_stays = list((filter(str.isdigit, unit_stays)))
    for stay_id in (unit_stays):
        df_file = os.path.join(output_path, str(stay_id), 'timeseries.csv')
        all_filenames.append(df_file)

    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_csv.to_csv(os.path.join(output_path, 'all_data.csv'), index=False)

#Convert categorical variables into number from 0 to 429
def prepare_categorical_variables(root_dir):
    columns_ord = [ 'patientunitstayid', 'itemoffset',
    'Eyes', 'Motor', 'GCS Total', 'Verbal',
    'ethnicity', 'gender','apacheadmissiondx',
    'FiO2','Heart Rate', 'Invasive BP Diastolic',
    'Invasive BP Systolic', 'MAP (mmHg)',  'O2 Saturation',
    'Respiratory Rate', 'Temperature (C)', 'admissionheight',
    'admissionweight', 'age', 'glucose', 'pH',
    'hospitaladmitoffset', 
    'hospitaldischargestatus','unitdischargeoffset',
    'unitdischargestatus']
    all_df = pd.read_csv(os.path.join(root_dir, 'all_data.csv'))
  
    all_df = all_df[all_df.gender != 0] #unknown gender is dropped
    all_df = all_df[all_df.hospitaldischargestatus != 2] #unknown hospital discharge is dropped
    all_df = all_df[columns_ord]

    all_df.apacheadmissiondx = all_df.apacheadmissiondx.astype(int)
    all_df.ethnicity = all_df.ethnicity.astype(int)
    all_df.gender = all_df.gender.astype(int)
    all_df['GCS Total'] = all_df['GCS Total'].astype(int)
    all_df['Eyes'] = all_df['Eyes'].astype(int)
    all_df['Motor'] = all_df['Motor'].astype(int)
    all_df['Verbal'] = all_df['Verbal'].astype(int)
    all_df.apacheadmissiondx = all_df.apacheadmissiondx + 1
    all_df.ethnicity = all_df.ethnicity + 1
    dxmax = all_df.apacheadmissiondx.max()
    etmax = all_df.ethnicity.max()
    gemax = all_df.gender.max()
    totmax = all_df['GCS Total'].max()
    eyemax = all_df['Eyes'].max()
    motmax = all_df['Motor'].max()
    vermax = all_df['Verbal'].max()
    all_df.ethnicity = all_df.ethnicity + dxmax
    all_df.gender = all_df.gender + dxmax+etmax
    all_df['GCS Total'] = all_df['GCS Total'] +dxmax+etmax+gemax
    all_df['Eyes'] = all_df['Eyes'] +dxmax+etmax+gemax+totmax
    all_df['Motor'] = all_df['Motor'] +dxmax+etmax+gemax+totmax+eyemax
    all_df['Verbal'] = all_df['Verbal'] +dxmax+etmax+gemax+totmax+eyemax+motmax
    return all_df

#Decompensation
def filter_decom_data(all_df):
    dec_cols = ['patientunitstayid', 'itemoffset', 'apacheadmissiondx', 'ethnicity', 'gender',
    'GCS Total', 'Eyes', 'Motor', 'Verbal',
    'admissionheight', 'admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
    'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
    'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH',
    'unitdischargestatus']
    # all_df = all_df[all_df.gender != 0]
    # all_df = all_df[all_df.hospitaldischargestatus!=2]
    all_df['RLOS'] = np.nan
    all_df['unitdischargeoffset'] = all_df['unitdischargeoffset'] / (1440)
    all_df['itemoffsetday'] = (all_df['itemoffset'] / 24)
    all_df['RLOS'] = (all_df['unitdischargeoffset'] - all_df['itemoffsetday'])
    all_df.drop(columns='itemoffsetday', inplace=True)
    all_dec = all_df[all_df["unitdischargestatus"] != 2]
    all_dec = all_dec[all_dec['itemoffset'] > 0]
    all_dec = all_dec[(all_dec['unitdischargeoffset'] > 1) & (all_dec['RLOS'] > 0)]
    all_dec = all_dec[dec_cols]
    return all_dec

# Labeling the decompensation task
def label_decompensation(all_dec):
    all_dec["temp_y"] = np.nan
    all_dec["temp_y"] = all_dec["itemoffset"] - 48
    all_dec['count_max'] = all_dec.groupby(['patientunitstayid'])['temp_y'].transform(max)
    all_dec["label_24"] = np.nan
    all_dec.loc[all_dec['itemoffset'] < all_dec['count_max'], "label_24"] = 0
    all_dec.loc[all_dec['itemoffset'] >= all_dec['count_max'], "label_24"] = all_dec['unitdischargestatus']
    all_dec["unitdischargestatus"] = all_dec["label_24"]
    all_dec.drop(columns=['temp_y', 'count_max', 'label_24'], inplace=True)
    all_dec.unitdischargestatus = all_dec.unitdischargestatus.astype(int)
    return all_dec          

def embedding(root_dir):
    all_df = prepare_categorical_variables(root_dir)
    return all_df


def df_to_list(df):
    grp_df  = df.groupby('patientunitstayid')
    df_arr = []
    for idx, frame in grp_df:
        df_arr.append(frame)
    
    return df_arr


def normalize_data_dec(config, data, train_idx, test_idx):

    train = data[data['patientunitstayid'].isin(train_idx)]
    test  = data[data['patientunitstayid'].isin(test_idx)]    
    col_used = ['patientunitstayid']

    if config.num and config.cat:
       col_used += config.dec_cat 
       col_used += config.dec_num
       col_used += ['unitdischargestatus']
    
    elif config.num:
        col_used += config.dec_num
        col_used += ['unitdischargestatus']
    
    elif config.cat:
        col_used += config.dec_cat
        col_used += ['unitdischargestatus']
        train = df_to_list(train[col_used])
        test = df_to_list(test[col_used])
        train, nrows_train = pad(train)
        test, nrows_test = pad(test)
        return (train, nrows_train), (test, nrows_test)

    train = train[col_used]
    test = test[col_used]
    
    cols_normalize = ['admissionheight','admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
       'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
       'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']
    feat_train_minmax = train[cols_normalize]
    scaler_minmax = MinMaxScaler(feature_range=(-1, 1), copy=True).fit(feat_train_minmax.values)
    feat_train_minmax = scaler_minmax.transform(feat_train_minmax.values)

    train[cols_normalize] = feat_train_minmax

    feat_test_minmax = test[cols_normalize]
    feat_test_minmax = scaler_minmax.transform(feat_test_minmax.values)
    test[cols_normalize] = feat_test_minmax

    train = df_to_list(train)
    test = df_to_list(test)

    train, nrows_train = pad(train)
    test, nrows_test = pad(test)

    return (train, nrows_train), (test, nrows_test)

def pad(data, max_len=200):
    padded_data = []
    nrows = []
    for item in data:
        tmp = np.zeros((max_len, item.shape[1]))
        tmp[:item.shape[0], :item.shape[1]] = item
        padded_data.append(tmp)
        nrows.append(item.shape[0])
    padded_data = np.array(padded_data)
    
    return padded_data, nrows

# Mortality
def filter_mortality_data(all_df):
    all_df = all_df[all_df.gender != 0]
    all_df = all_df[all_df.hospitaldischargestatus!=2]
    all_df['unitdischargeoffset'] = all_df['unitdischargeoffset']/(1440)
    all_df['itemoffsetday'] = (all_df['itemoffset']/24)
    all_df.drop(columns='itemoffsetday',inplace=True)
    mort_cols = ['patientunitstayid', 'itemoffset', 'apacheadmissiondx', 'ethnicity','gender',
                'GCS Total', 'Eyes', 'Motor', 'Verbal',
                'admissionheight','admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
                'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
                'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH',
                'unitdischargeoffset','hospitaldischargestatus']

    all_mort = all_df[mort_cols]
    all_mort = all_mort[all_mort['unitdischargeoffset'] >=2]
    all_mort = all_mort[all_mort['itemoffset']> 0]
    return all_mort



def normalize_data_mort(config, data, train_idx, test_idx):

    train = data[data['patientunitstayid'].isin(train_idx)]
    test  = data[data['patientunitstayid'].isin(test_idx)]
    col_used = ['patientunitstayid']

    if config.num and config.cat:
        col_used += config.dec_cat 
        col_used += config.dec_num
        col_used += ['hospitaldischargestatus']
    
    elif config.num:
        col_used += config.dec_num
        col_used += ['hospitaldischargestatus']
    
    elif config.cat:
        col_used += config.dec_cat
        col_used += ['hospitaldischargestatus']
        train = df_to_list(train[col_used])
        test = df_to_list(test[col_used])
        train, nrows_train = pad(train)
        test, nrows_test = pad(test)
        return (train, nrows_train), (test, nrows_test)
    
    train = train[col_used]
    test = test[col_used]

    cols_normalize = ['admissionheight','admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
       'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
       'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']
    feat_train_minmax = train[cols_normalize]
    scaler_minmax = MinMaxScaler(feature_range=(-1, 1), copy=True).fit(feat_train_minmax.values)
    feat_train_minmax = scaler_minmax.transform(feat_train_minmax.values)

    train[cols_normalize] = feat_train_minmax

    feat_test_minmax = test[cols_normalize]
    feat_test_minmax = scaler_minmax.transform(feat_test_minmax.values)
    test[cols_normalize] = feat_test_minmax

    train = df_to_list(train)
    test = df_to_list(test)

    train, nrows_train = pad(train)
    test, nrows_test = pad(test)

    return (train, nrows_train), (test, nrows_test)

# Phenotyping
def normalize_data_phe(config, data, train_idx, test_idx):
    col_used = ['patientunitstayid']
    train = data[data['patientunitstayid'].isin(train_idx)]
    test  = data[data['patientunitstayid'].isin(test_idx)]

    if config.num and config.cat:
        col_used += config.dec_cat 
        col_used += config.dec_num
        col_used += config.col_phe

    elif config.num :
        col_used += config.dec_num
        col_used += config.col_phe

    elif config.cat:
        col_used += config.dec_cat
        col_used += config.col_phe
        train = df_to_list(train[col_used])
        test = df_to_list(test[col_used])
        train, nrows_train = pad(train)
        test, nrows_test = pad(test)
        return (train, nrows_train), (test, nrows_test)
    
    train = train[col_used]
    test = test[col_used]
    cols_normalize = ['admissionheight','admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
       'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
       'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']
    feat_train_minmax = train[cols_normalize]
    scaler_minmax = MinMaxScaler(feature_range=(-1, 1), copy=True).fit(feat_train_minmax.values)
    feat_train_minmax = scaler_minmax.transform(feat_train_minmax.values)

    train[cols_normalize] = feat_train_minmax

    feat_test_minmax = test[cols_normalize]
    feat_test_minmax = scaler_minmax.transform(feat_test_minmax.values)
    test[cols_normalize] = feat_test_minmax

    train = df_to_list(train)
    test = df_to_list(test)

    train, nrows_train = pad(train)
    test, nrows_test = pad(test)

    return (train, nrows_train), (test, nrows_test)

def filter_phenotyping_data(all_df):
    all_df = all_df[all_df.gender != 0]
    all_df = all_df[all_df.hospitaldischargestatus!=2]
    all_df['RLOS'] = np.nan
    phen_cols = ['patientunitstayid', 'itemoffset', 'apacheadmissiondx', 'ethnicity','gender',
                'GCS Total', 'Eyes', 'Motor', 'Verbal',
                'admissionheight','admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
                'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
                'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH','RLOS']
    all_df['unitdischargeoffset'] = all_df['unitdischargeoffset'] / (1440)
    all_df['itemoffsetday'] = (all_df['itemoffset'] / 24)
    all_df['RLOS'] = (all_df['unitdischargeoffset'] - all_df['itemoffsetday'])
    all_df.drop(columns='itemoffsetday', inplace=True)
    all_df = all_df[phen_cols]
    return all_df


def read_diagnosis_table(eicu_path):
    diag = dataframe_from_csv(os.path.join(eicu_path, 'diagnosis.csv'), index_col=False)
    diag = diag[diag["diagnosisoffset"] > 0]
    diag = diag[['patientunitstayid', 'activeupondischarge', 'diagnosisoffset',
                'diagnosisstring', 'icd9code']]
    diag = diag[diag['icd9code'].notnull()]
    tes = diag['icd9code'].str.split(pat=",", expand=True, n=1)
    labels_name = ["Shock","Septicemia","Respiratory failure","Pneumonia","Pleurisy",
              "upper respiratory","lower respiratory","Other liver diseases",
              "Hypertension with complications","Gastrointestinal hem",
              "Fluid disorders","Essential hypertension","lipid disorder",
              "DM without complication","DM with complications",
              "Coronary athe","CHF", "Conduction disorders","Complications of surgical",
              "COPD", "CKD", "Cardiac dysrhythmias","Acute myocardial infarction",
               "Acute cerebrovascular disease","Acute and unspecified renal failure"]
    diag['icd0'] = np.nan
    diag['icd0'] = tes
    diag['icd'] = np.nan
    diag['icd'] = diag['icd0'].str.replace('.', '')
    diag = diag.reindex(columns=diag.columns.tolist() + labels_name)
    diag[labels_name] = np.nan
    return diag

def diag_labels(diag):
    import json
    codes = json.load(open('phen_code.json'))
    diag.loc[diag['icd'].isin(codes['septicemia']), 'Septicemia'] = 1
    diag.loc[diag['icd'].isin(codes['Shock']), 'Shock'] = 1
    diag.loc[diag['icd'].isin(codes['Compl_surgical']), 'Complications of surgical'] = 1
    diag.loc[diag['icd'].isin(codes['ckd']), 'CKD'] = 1
    diag.loc[diag['icd'].isin(codes['renal_failure']), 'Acute and unspecified renal failure'] = 1

    diag.loc[diag['icd'].isin(codes['Gastroint_hemorrhage']), 'Gastrointestinal hem'] = 1
    diag.loc[diag['icd'].isin(codes['Other_liver_dis']), 'Other liver diseases'] = 1
    diag.loc[diag['icd'].isin(codes['upper_respiratory']), 'upper respiratory'] = 1
    diag.loc[diag['icd'].isin(codes['lower_respiratory']), 'lower respiratory'] = 1
    diag.loc[diag['icd'].isin(codes['Resp_failure']), 'Respiratory failure'] = 1

    diag.loc[diag['icd'].isin(codes['Pleurisy']), 'Pleurisy'] = 1
    diag.loc[diag['icd'].isin(codes['COPD']), 'COPD'] = 1
    diag.loc[diag['icd'].isin(codes['Pneumonia']), 'Pneumonia'] = 1
    diag.loc[diag['icd'].isin(codes['Acute_cerebrovascular']), 'Acute cerebrovascular disease'] = 1
    diag.loc[diag['icd'].isin(codes['Congestive_hf']), 'CHF'] = 1

    diag.loc[diag['icd'].isin(codes['Cardiac_dysr']), 'Cardiac dysrhythmias'] = 1
    diag.loc[diag['icd'].isin(codes['Conduction_dis']), 'Conduction disorders'] = 1
    diag.loc[diag['icd'].isin(codes['Coronary_ath']), 'Coronary athe'] = 1
    diag.loc[diag['icd'].isin(codes['myocar_infarction']), 'Acute myocardial infarction'] = 1
    diag.loc[diag['icd'].isin(codes['hypercomp']), 'Hypertension with complications'] = 1

    diag.loc[diag['icd'].isin(codes['essehyper']), 'Essential hypertension'] = 1
    diag.loc[diag['icd'].isin(codes['fluiddiso']), 'Fluid disorders'] = 1
    diag.loc[diag['icd'].isin(codes['lipidmetab']), 'lipid disorder'] = 1
    diag.loc[diag['icd'].isin(codes['t2dmcomp']), 'DM with complications'] = 1
    diag.loc[diag['icd'].isin(codes['t2dmwocomp']), 'DM without complication'] = 1
    return diag

def diag_df_to_numpy(df,diag_g):
    df_grpd = df.groupby('patientunitstayid')
    idx = []
    df_array = []
    df_label = pd.DataFrame()
    for idx, frame in df_grpd:
        idts.append(idx)
        test_np.append(frame)
        df_label = pd.concat([df_label, diag_g[diag_g.patientunitstayid == idx]])

    return df_array,df_label



#Remaining Length of Stay

def filter_rlos_data(all_df):
    los_cols = ['patientunitstayid', 'itemoffset', 'apacheadmissiondx', 'ethnicity', 'gender',
            'GCS Total', 'Eyes', 'Motor', 'Verbal',
            'admissionheight', 'admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)',
            'Invasive BP Diastolic', 'Invasive BP Systolic', 'O2 Saturation',
            'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH',
            'unitdischargeoffset', 'RLOS']

    import pdb;pdb.set_trace()

    all_df = all_df[all_df.gender != 0]
    all_df = all_df[all_df.hospitaldischargestatus!=2]
    all_df['RLOS'] = np.nan
    all_df['unitdischargeoffset'] = all_df['unitdischargeoffset'] / (1440)
    all_df['itemoffsetday'] = (all_df['itemoffset'] / 24)

    all_df['RLOS'] = (all_df['unitdischargeoffset'] - all_df['itemoffsetday'])
    all_df.drop(columns='itemoffsetday', inplace=True)
    
    all_los = all_df[los_cols]
    all_los = all_los[all_los['itemoffset'] > 0]
    all_los = all_los[(all_los['unitdischargeoffset'] > 0) & (all_los['RLOS'] > 0)]
    all_los = all_los.round({'RLOS': 2})
    return all_los



def normalize_data_rlos(config, data, train_idx, test_idx):
    col_used = ['patientunitstayid']
    train = data[data['patientunitstayid'].isin(train_idx)]
    test  = data[data['patientunitstayid'].isin(test_idx)]

    if config.num and config.cat:
        col_used += config.dec_cat 
        col_used += config.dec_num
        col_used += ['RLOS']
    
    elif config.num:
        col_used += config.dec_num
        col_used += ['RLOS']  
    
    elif config.cat:
        col_used += config.dec_cat
        col_used += ['RLOS']
        train = df_to_list(train[col_used])
        test = df_to_list(test[col_used])
        train, nrows_train = pad(train)
        test, nrows_test = pad(test)
        return (train, nrows_train), (test, nrows_test)
    
    train = train[col_used]
    test = test[col_used]
    cols_normalize = config.dec_num
    feat_train_minmax = train[cols_normalize]
    scaler_minmax = MinMaxScaler(feature_range=(-1, 1), copy=True).fit(feat_train_minmax.values)
    feat_train_minmax = scaler_minmax.transform(feat_train_minmax.values)

    train[cols_normalize] = feat_train_minmax

    feat_test_minmax = test[cols_normalize]
    feat_test_minmax = scaler_minmax.transform(feat_test_minmax.values)
    test[cols_normalize] = feat_test_minmax

    train = df_to_list(train)
    test = df_to_list(test)

    train, nrows_train = pad(train)
    test, nrows_test = pad(test)

    return (train, nrows_train), (test, nrows_test)