from config import Config
import argparse
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
from models import data_reader
from sklearn.metrics import roc_curve, auc,confusion_matrix,average_precision_score,matthews_corrcoef
from scipy import interp
from models import evaluation
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



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

    skf = KFold(n_splits=2)
    for train_idx, test_idx in skf.split(all_idx):
        train_idx = all_idx[train_idx]
        test_idx = all_idx[test_idx]
        if config.num and config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_dec(config, train, test,numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        elif config.num and not config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_dec(config, train, test,numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        elif not config.num and config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_dec(config, train, test, numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        model = network(200, numerical=config.num, categorical=config.cat)

        history = model.fit_generator(train_gen,steps_per_epoch=25,
                            epochs=config.epochs,verbose=1,shuffle=True)
        if config.num and config.cat:
            probas_dec = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
        elif config.num and not config.cat:
            probas_dec = model.predict([X_test])
        elif not config.num and config.cat:
            probas_dec = model.predict([X_test])

        Y_test, probas_dec = evaluation.decompensation_metrics(Y_test,probas_dec,max_time_step_test)
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
    print("====================Decompensation================")
    print("Mean AUC{0:0.3f} +- STD{1:0.3f}".format(mean_auc_dec,std_auc_dec))
    print("PPV: {0:0.3f}".format(np.mean(ppvs_dec)))
    print("NPV: {0:0.3f}".format(np.mean(npvs_dec)))
    print("AUCPR:{0:0.3f}".format(np.mean(aucprs_dec)))
    print("MCC: {0:0.3f}".format(np.mean(mccs_dec)))
    print("Spec@90: {0:0.3f}".format(np.mean(specat90_dec)))


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
    skf = KFold(n_splits=2)

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
        
        model = network(input_size=config.mort_window, numerical=config.num, categorical=config.cat)

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
        

#Phenotyping
def train_phen(config):
    import numpy as np
    from models.models import network_phenotyping as network
    from data_extraction.utils import normalize_data_phe as normalize_data
    from data_extraction.data_extraction_phenotyping import data_extraction_phenotyping
    df_data, df_label = data_extraction_phenotyping(config)
    df_data = df_data.merge(df_label.drop(columns=['itemoffset']), on='patientunitstayid')
    all_idx = np.array(list(df_data['patientunitstayid'].unique()))   
   
    phen_auc  = []
    phen_aucs = []
    skf = KFold(n_splits=2)
    for train_idx, test_idx in skf.split(all_idx):
        train_idx = all_idx[train_idx]
        test_idx = all_idx[test_idx]

        if config.num and config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_phe(config, train, test,numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        elif config.num and not config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_phe(config, train, test,numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        elif not config.num and config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_phe(config, train, test, numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        
        model = network(200, numerical=config.num, categorical=config.cat)
        history = model.fit_generator(train_gen,steps_per_epoch=25,
                            epochs=config.epochs,verbose=1,shuffle=True)

        if config.num and config.cat:
            probas_phen = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
        elif config.num and not config.cat:
            probas_phen = model.predict([X_test])
        elif not config.num and config.cat:
            probas_phen = model.predict([X_test])

        phen_auc = evaluation.multi_label_metrics(Y_test,probas_phen)
        phen_aucs.append(phen_auc)
    aucs = np.mean(np.array(phen_aucs),axis=0)
    for i in range(len(config.col_phe)):
        print("{0} : {1:0.3f}".format(config.col_phe[i],aucs[i]))

# Remaining length of stay

def train_rlos(config):
    from models.models import network_los as network
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
        elif config.num and not config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_rlos(config, train, test,numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        elif not config.num and config.cat:
            train, test = normalize_data(config, df_data,train_idx, test_idx, cat=config.cat, num=config.num)
            train_gen, train_steps, (X_test, Y_test), max_time_step_test = data_reader.data_reader_for_model_rlos(config, train, test, numerical=config.num, categorical=config.cat,  batch_size=1024, val=False)
        
        model = network(200, numerical=config.num, categorical=config.cat)

        history = model.fit_generator(train_gen,steps_per_epoch=25,
                            epochs=config.epochs,verbose=1,shuffle=True)
        if config.num and config.cat:
            probas_rlos = model.predict([X_test[:,:,7:],X_test[:,:,:7]])
        elif config.num and not config.cat:
            probas_rlos = model.predict([X_test])
        elif not config.num and config.cat:
            probas_rlos = model.predict([X_test])
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
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1
    session = tf.Session(config=tf_config)
    K.set_session(session)


    config = Config()

    np.random.seed(config.seed)

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