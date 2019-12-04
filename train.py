# from models.data_reader import *
# from models.models import *
import argparse
# from models.data_reader import data_reader_decompensation as read_data
# from models.models import network_decompensation as network
from keras import backend as K
import tensorflow as tf
from config import Config
from sklearn.model_selection import KFold
import numpy as np
from models import data_reader
from sklearn.metrics import roc_curve, auc,confusion_matrix,average_precision_score,matthews_corrcoef
from scipy import interp


import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


#Define method for ROC curve:

#Decompensation
def train_dec(config):
    from models.models import network_decompensation as network
    from data_extraction.utils import normalize_data_dec as normalize_data
    from data_extraction.data_extraction_decompensation import data_extraction_decompensation
    df_data = data_extraction_decompensation(config)

    cvscores_dec = []
    tprs_dec = []
    aucs_dec = []
    mean_fpr_dec = np.linspace(0, 1, 100)
    i_dec = 0
    ppvs_dec = []
    npvs_dec = []
    aucprs_dec = []
    mccs_dec = []
    specat90_dec = []

    all_idx = np.array(list(df_data['patientunitstayid'].unique()))

    skf = KFold(n_splits=5)
    for train_idx, test_idx in skf.split(all_idx):
        train_idx = all_idx[train_idx]
        test_idx = all_idx[test_idx]

        train, test = normalize_data(config, df_data,train_idx, test_idx, cat=True, num=True)

        train_gen, train_steps, (X_test, Y_test), max_time_step = data_reader.data_reader_for_model_dec(train, test, batch_size=1024, val=False)
        # import pdb
        # pdb.set_trace()
        model = network(max_time_step, numerical=config.num, categorical=config.cat)
        model.summary()

        history = model.fit_generator(train_gen,steps_per_epoch=25,
                            epochs=config.epochs,verbose=1,shuffle=True)

        probas_dec = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
        probas_dec = np.squeeze(probas_dec,axis=-1)
        import pdb
        pdb.set_trace()
        fpr_dec, tpr_dec, thresholds = roc_curve(Y_test, probas_dec)
        tprs_dec.append(interp(mean_fpr_dec, fpr_dec, tpr_dec))
        tprs_dec[-1][0] = 0.0
        roc_auc_dec = auc(fpr_dec, tpr_dec)
        aucs_dec.append(roc_auc_dec)
        TN,FP,FN,TP = confusion_matrix(Y_test,probas_dec.round()).ravel()
        PPV = TP/(TP+FP)
        NPV = TN/(TN+FN)
        ppvs_dec.append(PPV)
        npvs_dec.append(NPV)
        average_precision_dec = average_precision_score(Y_test,probas_dec)
        aucprs_dec.append(average_precision_dec)
        mccs_dec.append(matthews_corrcoef(Y_test, probas_dec.round()))
        specat90_dec.append(1-fpr_dec[tpr_dec>=0.90][0])
    mean_tpr_dec = np.mean(tprs_dec, axis=0)
    mean_tpr_dec[-1] = 1.0
    mean_auc_dec = auc(mean_fpr_dec, mean_tpr_dec)
    std_auc_dec = np.std(aucs_dec)
    print("========================")
    print("Mean AUC{0:0.3f} +- STD{1:0.3f}".format(mean_auc_dec,std_auc_dec))
    print("PPV: {0:0.3f}".format(np.mean(ppvs_dec)))
    print("NPV: {0:0.3f}".format(np.mean(npvs_dec)))
    print("AUCPR:{0:0.3f}".format(np.mean(aucprs_dec)))
    print("MCC: {0:0.3f}".format(np.mean(mccs_dec)))
    print("Spec@90: {0:0.3f}".format(np.mean(specat90_dec)))

        # [test_file['X_noncat'], test_file['X_cat']

        #Y_pred = model.evaluate

        #pass to Roc curve function

        #add to list
    
    #print average

#Mortality
def train_mort(config):
    
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

    from models.models import network_mortality as network
    from data_extraction.data_extraction_mortality import data_extraction_mortality
    from data_extraction.utils import normalize_data_mort as normalize_data

    df_data = data_extraction_mortality(config)
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))
    skf = KFold(n_splits=5)

    for train_idx, test_idx in skf.split(all_idx):
        train_idx = all_idx[train_idx]
        test_idx = all_idx[test_idx]

        train, test = normalize_data(config, df_data,train_idx, test_idx, cat=True, num=True)

        train_gen, train_steps, (X_test, Y_test), max_time_step = data_reader.data_reader_for_model_mort(train, test, batch_size=1024, val=False)

        model = network(max_time_step, numerical=config.num, categorical=config.cat)
        # model.summary()

        history = model.fit_generator(train_gen,steps_per_epoch=25,
                            epochs=config.epochs,verbose=1,shuffle=True)

        probas_mort = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
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
        
        #auc / roc --> test:

#Phenotyping
def train_phen(config):
    from models.models import network_phenotyping as network
    from data_extraction.utils import normalize_data_phe as normalize_data
    from data_extraction.data_extraction_phenotyping import data_extraction_phenotyping
    df_data, df_label = data_extraction_phenotyping(config)
    df_data = df_data.merge(df_label.drop(columns=['itemoffset']), on='patientunitstayid')

    # import pdb
    # pdb.set_trace()
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))

    skf = KFold(n_splits=5)
    for train_idx, test_idx in skf.split(all_idx):
        train_idx = all_idx[train_idx]
        test_idx = all_idx[test_idx]

        train, test = normalize_data(config, df_data, train_idx, test_idx, cat=True, num=True)

        train_gen, train_steps, (X_test, Y_test), max_time_step = data_reader.data_reader_for_model_phe(config, train, test, batch_size=1024, val=False)

        model = network(max_time_step, numerical=config.num, categorical=config.cat)
        model.summary()

        history = model.fit_generator(train_gen,steps_per_epoch=25,
                            epochs=config.epochs,verbose=1,shuffle=True)


#Remaining length of stay
def train_rlos(config):
    from models.models import network_decompensation as network
    from data_extraction.utils import normalize_data_rlos as normalize_data
    from data_extraction.data_extraction_rlos import data_extraction_rlos
    df_data = data_extraction_rlos(config)
    import pdb
    pdb.set_trace()
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))

    skf = KFold(n_splits=5)
    for train_idx, test_idx in skf.split(all_idx):
        train_idx = all_idx[train_idx]
        test_idx = all_idx[test_idx]

        train, test = normalize_data(config, df_data,train_idx, test_idx, cat=True, num=True)

        train_gen, train_steps, (X_test, Y_test), max_time_step = data_reader.data_reader_for_model_dec(train, test, batch_size=1024, val=False)

        model = network(max_time_step, numerical=config.num, categorical=config.cat)
        model.summary()

        history = model.fit_generator(train_gen,steps_per_epoch=25,
                            epochs=config.epochs,verbose=1,shuffle=True)

        
        #auc / roc --> test:


def main():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1
    session = tf.Session(config=tf_config)
    K.set_session(session)


    config = Config()

    if config.task == 'dec':
        train_dec(config)
    elif config.task =='mort':
        train_mort(config)
    elif config.task == 'phen':
        train_phen(config)
    elif config.task =='rlos':
        train_rlos(config)
    else:
        print('Invalid task name')

if __name__ == "__main__":
    main()