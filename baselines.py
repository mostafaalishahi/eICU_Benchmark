from __future__ import absolute_import
from __future__ import print_function
import os
from config import Config
import argparse
from sklearn.model_selection import KFold
import numpy as np
from models import data_reader
from sklearn.metrics import roc_curve, auc,confusion_matrix,average_precision_score,matthews_corrcoef
from scipy import interp
from models import evaluation
import sys
import json
import numpy as np
import random
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from keras.layers import BatchNormalization, Dropout, Dense, TimeDistributed, Masking, Activation, Input, Reshape, Embedding
from keras import optimizers,regularizers
from keras.callbacks import ModelCheckpoint

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

##############################################Performance Metrics########################################################
def get_optimizer(lr=0.0005):
    adam = optimizers.Adam(lr=lr)
    return adam
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
#####################################################################################################

def base_mort(input_size, catg_len=429, embedding_dim=5, numerical=True, categorical=True,ann=False):
    if numerical and categorical:
        input1 = Input(shape=(input_size, 13))
        input2 = Input(shape=(input_size, 7))
        x2 = Embedding(catg_len, embedding_dim)(input2)
        x2 = Reshape((int(x2.shape[1]),int(x2.shape[2]*x2.shape[3])))(x2)
        inp = keras.layers.Concatenate(axis=-1)([input1, x2])
    elif numerical:
        input1 = Input(shape=(input_size, 13))
        inp = input1
    elif categorical:
        input1 = Input(shape=(input_size, 7))
        x1 = Embedding(catg_len, embedding_dim)(input1)
        inp = Reshape((int(x1.shape[1]),int(x1.shape[2]*x1.shape[3])))(x1)
    mask = Reshape((int(x2.shape[1])*int(x2.shape[2]+input1.shape[2]),))(inp)
    if ann:
        mask = keras.layers.Dense(64,activation='relu')(mask)
    out = keras.layers.Dense(1,activation="sigmoid")(mask)
    if numerical and categorical:
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    else:
        model = keras.models.Model(inputs=input1, outputs=out)
    adam = get_optimizer(lr=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=adam ,metrics=[f1,sensitivity, specificity,'accuracy'])
    return model

def base_rlos(input_size, catg_len=429, embedding_dim=5, numerical=True, categorical=True,ann=False):

    if numerical and categorical:
        input1 = Input(shape=(input_size, 13))
        input2 = Input(shape=(input_size, 7))
        x2 = Embedding(catg_len, embedding_dim)(input2)
        x2 = Reshape((int(x2.shape[1]),int(x2.shape[2]*x2.shape[3])))(x2)
        inp = keras.layers.Concatenate(axis=-1)([input1, x2])
    elif numerical:
        input1 = Input(shape=(input_size, 13))
        inp = input1
    elif categorical:
        input1 = Input(shape=(input_size, 7))
        x1 = Embedding(catg_len, embedding_dim)(input1)
        inp = Reshape((int(x1.shape[1]),int(x1.shape[2]*x1.shape[3])))(x1)
    mask = Reshape((int(x2.shape[1])*int(x2.shape[2]+input1.shape[2]),))(inp)
    if ann:
        mask = keras.layers.Dense(64,activation='relu')(mask)
    out = TimeDistributed(Dense(1,activation="relu"))(mask)
    if numerical and categorical:
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    else:
        model = keras.models.Model(inputs=input1, outputs=out)
    adam = get_optimizer(lr=0.005)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
    return model

def base_dec(input_size, catg_len=429, embedding_dim=5, numerical=True, categorical=True,ann=False):
    if numerical and categorical:
        input1 = Input(shape=(input_size, 13))
        input2 = Input(shape=(input_size, 7))
        x2 = Embedding(catg_len, embedding_dim)(input2)
        x2 = Reshape((int(x2.shape[1]),int(x2.shape[2]*x2.shape[3])))(x2)
        inp = keras.layers.Concatenate(axis=-1)([input1, x2])
    elif numerical:
        input1 = Input(shape=(input_size, 13))
        inp = input1
    elif categorical:
        input1 = Input(shape=(input_size, 7))
        x1 = Embedding(catg_len, embedding_dim)(input1)
        inp = Reshape((int(x1.shape[1]),int(x1.shape[2]*x1.shape[3])))(x1)
    mask = Reshape((int(x2.shape[1])*int(x2.shape[2]+input1.shape[2]),))(inp)
    if ann:
        mask = keras.layers.Dense(64,activation='relu')(mask)
    out = TimeDistributed(Dense(1,activation="relu"))(mask)
    if numerical and categorical:
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    else:
        model = keras.models.Model(inputs=input1, outputs=out)
    adam = get_optimizer(lr=0.005)
    model.compile(loss="binary_crossentropy", optimizer=adam ,metrics=[f1,sensitivity, specificity,'accuracy'])
    return model

def transform_hospital_discharge_status(status_series):
    global h_s_map
    return {'actualhospitalmortality': status_series.fillna('').apply(
        lambda s: h_s_map[s] if s in h_s_map else h_s_map[''])}

def mort_apache(config):
    h_s_map = {'ALIVE': 0, 'EXPIRED': 1, '': 2, 'NaN': 2}
    apache.update(transform_hospital_discharge_status(apache.actualhospitalmortality))
    apache = pd.read_csv(os.path.join(config.eicu_dir, 'apachePatientResult.csv'), index_col=False)
    apache = apache[apache.apacheversion=="IVa"]
    apache.loc[apache['predictedhospitalmortality']<=0,'predictedhospitalmortality']=0
    apache.reset_index(inplace=True)
    y_true = apache.actualhospitalmortality
    y_pred = apache.predictedhospitalmortality
    return y_pred,y_true

def mort(config):
    cvscores_mort = []
    tprs_mort = []
    aucs_mort = []
    mean_fpr_mort = np.linspace(0, 1, 100)
    i_mort = 0
    ppvs_mort = []
    npvs_mort = []
    aucprs_mort = []
    mccs_mort = []
    specat90_mort = []

    from data_extraction.data_extraction_mortality import data_extraction_mortality
    from data_extraction.utils import normalize_data_mort as normalize_data

    df_data = data_extraction_mortality(config)
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))
    skf = KFold(n_splits=5)

    for train_idx, test_idx in skf.split(all_idx):
        train_idx = all_idx[train_idx]
        test_idx = all_idx[test_idx]

        if config.num and config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_mort(config, train, test,numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        elif config.num and not config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_mort(config, train, test,numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        elif not config.num and config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_mort(config, train, test, numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        
        model = base_mort(input_size=200, numerical=config.num, categorical=config.cat,ann=config.ann)

        history = model.fit_generator(train_gen,steps_per_epoch=25,
                            epochs=config.epochs,verbose=1,shuffle=True)
        
        if config.num and config.cat:
            probas_mort = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
        elif config.num and not config.cat:
            probas_mort = model.predict([X_test])
        elif not config.num and config.cat:
            probas_mort = model.predict([X_test])

        
        fpr_mort, tpr_mort, thresholds = roc_curve(Y_test, probas_mort)
        tprs_mort.append(interp(mean_fpr_mort, fpr_mort, tpr_mort))
        tprs_mort[-1][0] = 0.0
        roc_auc_mort = auc(fpr_mort, tpr_mort)
        aucs_mort.append(roc_auc_mort)
        TN,FP,FN,TP = confusion_matrix(Y_test,probas_mort.round()).ravel()
        PPV = TP/(TP+FP)
        NPV = TN/(TN+FN)

        ppvs_mort.append(PPV)
        npvs_mort.append(NPV)
        average_precision_mort = average_precision_score(Y_test,probas_mort)
        aucprs_mort.append(average_precision_mort)
        mccs_mort.append(matthews_corrcoef(Y_test, probas_mort.round()))
        specat90_mort.append(1-fpr_mort[tpr_mort>=0.90][0])

    mean_tpr_mort = np.mean(tprs_mort, axis=0)
    mean_tpr_mort[-1] = 1.0
    mean_auc_mort = auc(mean_fpr_mort, mean_tpr_mort)
    std_auc_mort = np.std(aucs_mort)

    print("===========================Mortality=============================")
    print("Mean AUC {0:0.3f} +- STD{1:0.3f}".format(mean_auc_mort,std_auc_mort))
    print("PPV: {0:0.3f}".format(np.mean(ppvs_mort)))
    print("NPV: {0:0.3f}".format(np.mean(npvs_mort)))
    print("AUCPR:{0:0.3f}".format(np.mean(aucprs_mort)))
    print("MCC: {0:0.3f}".format(np.mean(mccs_mort)))
    print("Spec@90: {0:0.3f}".format(np.mean(specat90_mort)))

def rlos(config):
    from data_extraction.utils import normalize_data_rlos as normalize_data
    from data_extraction.data_extraction_rlos import data_extraction_rlos
    df_data = data_extraction_rlos(config)
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))

    r2s= []
    mses = []
    maes = []
    skf = KFold(n_splits=2)
    for train_idx, test_idx in skf.split(all_idx):
        train_idx = all_idx[train_idx]
        test_idx = all_idx[test_idx]
        if config.num and config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_rlos(config, train, test,numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        # elif config.num and not config.cat:
        #     train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
        #     train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_rlos(config, train, test,numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        # elif not config.num and config.cat:
        #     train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
        #     train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_rlos(config, train, test, numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        
        model = base_dec(input_size=200, catg_len=429, embedding_dim=5, numerical=config.num, categorical=config.cat,ann=config.ann)

        history = model.fit_generator(train_gen,steps_per_epoch=25,
                            epochs=config.epochs,verbose=1,shuffle=True)
        if config.num and config.cat:
            probas_rlos = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
        # elif config.num and not config.cat:
        #     probas_rlos = model.predict([X_test])
        # elif not config.num and config.cat:
        #     probas_rlos = model.predict([X_test])
        r2,mse,mae = evaluation.regression_metrics(Y_test,probas_rlos,max_time_step_test)
        r2s.append(r2)
        mses.append(mse)
        maes.append(mae)

    meanr2s = np.mean(r2s)
    meanmses = np.mean(mses)
    meanmaes = np.mean(maes)

    stdr2s = np.std(r2s)
    stdmses = np.std(mses)
    stdmaes = np.std(maes)

    print("===========================RLOS=============================")
    print("R2 total: {0:0.3f} +- {1:0.3f} ".format(meanr2s,stdr2s))
    print("MSE total: {0:0.3f}  +- {1:0.3f}".format(meanmses,stdmses))
    print("MAE total:{0:0.3f}  +- {1:0.3f}".format(meanmaes,stdmaes))



def main():
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 1
    # session = tf.Session(config=tf_config)
    # K.set_session(session)
    config = Config()
    np.random.seed(config.seed)

    if config.task == 'dec':
        train_dec(config)
    elif config.task =='mort':
        mort(config)
    elif config.task =='rlos':
        rlos(config)
    else:
        print('Invalid task name')

if __name__ == "__main__":
    main()