import numpy as np
import random
from config import Config

config = Config()

def batch_generator(config, X, Y, batch_size=1024, rng=np.random.RandomState(0), train=True, phen=True):
        if train:
            while True:
                # data = list(zip(X, Y))
                all_index = list(range(X.shape[0]))
                while len(all_index) > (batch_size*0.2):
                    idx = rng.choice(all_index, int(batch_size))
                    x_batch = X[idx]
                    y_batch = Y[idx]
                    # data_selection = data[idx]
                    idx = list(set(all_index) - set(idx))
                    data_selection = list(zip(x_batch, y_batch))
                    random.shuffle(data_selection)
                    x_batch, y_batch = zip(*data_selection)

                    x_batch = np.array(x_batch)
                    y_batch = np.array(y_batch)

                    if not phen:
                        y_batch = np.expand_dims(y_batch, axis=-1)
                    
                    if config.num and config.cat:
                        x_nc = x_batch[:, :, 7:]
                        x_cat = x_batch[:,:, :7].astype(int)
                        if config.ohe:
                            one_hot = np.zeros((x_cat.shape[0], x_cat.shape[1], 429), dtype=np.int)
                            one_hot = (np.eye(429)[x_cat].sum(2) > 0).astype(int)
                            x_cat = one_hot
                        yield [x_nc, x_cat], y_batch
                    
                    else:
                        yield x_batch, y_batch
                                
        else:
            while True:
                X = np.array(X)
                Y = np.array(Y)
                if not phen:
                    Y = np.expand_dims(Y, axis=-1)
                for i in range(0, len(Y), batch_size):
                    st_idx = i
                    end_idx = st_idx + batch_size

                    if config.num and config.cat:
                        x_nc = X[:, :, 7:]
                        x_cat = X[:, :, :7]
                        yield [x_nc[st_idx:end_idx], x_cat[st_idx:end_idx]], Y[st_idx:end_idx]
                    else:
                        yield X[st_idx:end_idx], Y[st_idx:end_idx]


def read_data(config, train, test, val=False):
    nrows_train = train[1]
    nrows_test = test[1]

    if config.task == 'phe':
        n_labels = len(config.col_phe)
    elif config.task in ['dec', 'mort', 'rlos']:
        n_labels = 1

    X_train = train[0][:, :, 1:-n_labels] #column 0 is patient_id
    X_test = test[0][:, :, 1:-n_labels]

    if config.num and config.cat:        
        X_train = X_train[:,:,:-1]
        X_test = X_test[:,:,:-1]

    elif config.num:
        X_train = X_train[:,:,7:-1] 
        X_test = X_test[:,:,7:-1]

    elif config.cat:
        X_train = X_train[:,:,0:7]
        X_test = X_test[:,:,0:7]    

    Y_train = train[0][:, 0, -n_labels:]
    Y_test = test[0][:, 0, -n_labels:]    
    X_train = list(zip(X_train, nrows_train))

    if val:
        (X_train, Y_train), (X_val, Y_val) = train_test_split(X_train, Y_train, split_size=0.2)        
        X_val, nrows_val = zip(*X_val)

    X_train, nrows_train = zip(*X_train)
    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)
    X_train = np.array(X_train)

    train_gen = batch_generator(config, X_train, Y_train, batch_size=config.batch_size, train=True, phen=True)
    train_steps = np.ceil(len(X_train)/config.batch_size)

    if val:
        Y_val = Y_val.astype(int)        
        val_gen   = batch_generator(X_val, Y_val, batch_size=config.batch_size, train=False,phen=True)
        val_steps = np.ceil(len(X_val)/config.batch_size)

    max_time_step = nrows_test
    if val:
        return train_gen, train_steps, val_gen, val_steps, (X_test, Y_test), max_time_step

    return  train_gen, train_steps, (X_test, Y_test), max_time_step