import numpy as np
import random
from config import Config

config = Config()

def batch_generator(config,X, Y, batch_size=1024, rng=np.random.RandomState(0), numerical=config.num, categorical=config.cat, train=True,phen=True):
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
                    
                    if numerical and categorical:
                        x_nc = x_batch[:, :, 7:]
                        x_cat = x_batch[:,:, :7]
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

                    if numerical and categorical:
                        x_nc = X[:, :, 7:]
                        x_cat = X[:, :, :7]
                        yield [x_nc[st_idx:end_idx], x_cat[st_idx:end_idx]], Y[st_idx:end_idx]
                    else:
                        yield X[st_idx:end_idx], Y[st_idx:end_idx]


def data_reader_for_model_phe(config, train, test,numerical=config.num, categorical=config.cat,  batch_size=1024, val=False):
    nrows_train = train[1]
    nrows_test = test[1]

    if numerical and categorical:
        # import pdb
        # pdb.set_trace()
        X_train = train[0][:, :, 2:-len(config.col_phe)] #column 0 is patient_id and column 1 is time step,last columns are labels
        X_test = test[0][:, :, 2:-len(config.col_phe)]
        X_train = X_train[:,:,:-1] # last column is length of stay
        X_test = X_test[:,:,:-1] 
        # import pdb
        # pdb.set_trace()

    elif numerical and not categorical:
        X_train = train[0][:, :, 2:-len(config.col_phe)] #column 0 is patient_id and column 1 is time step,last columns are labels
        X_test = test[0][:, :, 2:-len(config.col_phe)]
        X_train = X_train[:,:,:-1] # last column is length of stay
        X_test = X_test[:,:,:-1] 
        X_train = X_train[:,:,7:]
        X_test = X_test[:,:,7:]

    elif not numerical and categorical:
        X_train = train[0][:, :, 2:-len(config.col_phe)] #column 0 is patient_id and column 1 is time step,last columns are labels
        X_test = test[0][:, :, 2:-len(config.col_phe)]
        X_train = X_train[:,:,:-1]
        X_test = X_test[:,:,:-1]
        X_train = X_train[:,:,0:7]
        X_test = X_test[:,:,0:7]
    
    Y_train = train[0][:, 0, -len(config.col_phe):]
    Y_test = test[0][:, 0, -len(config.col_phe):]
    X_train = list(zip(X_train, nrows_train))
   
    if val:
        (X_train, Y_train), (X_val, Y_val) = train_test_split(X_train, Y_train, split_size=0.2)        
        X_val, nrows_val = zip(*X_val)
    
    X_train, nrows_train = zip(*X_train)
    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)
    X_train = np.array(X_train)

    if numerical and categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=True,categorical=True, batch_size=batch_size, train=True,phen=True)
        train_steps = np.ceil(len(X_train)/batch_size)
    if numerical and not categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=True,categorical=False, batch_size=batch_size, train=True,phen=True)
        train_steps = np.ceil(len(X_train)/batch_size)
    if not numerical and categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=False,categorical=True, batch_size=batch_size, train=True,phen=True)
        train_steps = np.ceil(len(X_train)/batch_size)

    if val:
        Y_val = Y_val.astype(int)        
        val_gen   = batch_generator(X_val, Y_val, batch_size=batch_size, train=False,phen=True)
        val_steps = np.ceil(len(X_val)/batch_size)

    max_time_step = nrows_test
    if val:
        return train_gen, train_steps, val_gen, val_steps, (X_test, Y_test), max_time_step

    return  train_gen, train_steps, (X_test, Y_test), max_time_step


def data_reader_for_model_dec(config,train, test, numerical=config.num, categorical=config.cat, batch_size=1024, val=False):

    nrows_train = train[1]
    nrows_test = test[1]

    
    if numerical and categorical:
        X_train = train[0][:, :, 2:-1] #column 0 is patient_id and column 1 is time step
        X_test = test[0][:, :, 2:-1]

    elif numerical and not categorical:
        X_train = train[0][:, :, 2:-1] #column 0 is patient_id and column 1 is time step
        X_test = test[0][:, :, 2:-1]
        X_train = X_train[:,:,7:]
        X_test = X_test[:,:,7:]

    elif not numerical and categorical:
        X_train = train[0][:, :, 2:-1] #column 0 is patient_id and column 1 is time step
        X_test = test[0][:, :, 2:-1]
        X_train = X_train[:,:,0:7]
        X_test = X_test[:,:,0:7]
    
    Y_train = train[0][:, :, -1]
    Y_test = test[0][:, :, -1]

    X_train = list(zip(X_train, nrows_train))
    if val:
        (X_train, Y_train), (X_val, Y_val) = train_test_split(X_train, Y_train, split_size=0.2)        
        X_val, nrows_val = zip(*X_val)
    
    X_train, nrows_train = zip(*X_train)

    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)
    X_train = np.array(X_train)
    
    if numerical and categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=True,categorical=True, batch_size=batch_size, train=True,phen=False)
        train_steps = np.ceil(len(X_train)/batch_size)
    if numerical and not categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=True,categorical=False, batch_size=batch_size, train=True,phen=False)
        train_steps = np.ceil(len(X_train)/batch_size)
    if not numerical and categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=False,categorical=True, batch_size=batch_size, train=True,phen=False)
        train_steps = np.ceil(len(X_train)/batch_size)

    if val:
        Y_val = Y_val.astype(int)        
        val_gen   = batch_generator(config,X_val, Y_val,numerical=True,categorical=True, batch_size=batch_size, train=False,phen=False)
        val_steps = np.ceil(len(X_val)/batch_size)

    max_time_step = nrows_test
    if val:
        return train_gen, train_steps, val_gen, val_steps, (X_test, Y_test), max_time_step

    return  train_gen, train_steps, (X_test, Y_test), max_time_step

def data_reader_for_model_mort(config,train, test, numerical=config.num, categorical=config.cat,batch_size=1024, val=False):

    nrows_train = train[1]
    nrows_test = test[1]

    if numerical and categorical:
        import pdb
        pdb.set_trace()
        X_train = train[0][:, :, 2:-2] #column 0 is patient_id and column 1 is time step, last col is mort label, 2nd to last is los
        X_test = test[0][:, :, 2:-2]

    elif numerical and not categorical:
        X_train = train[0][:, :, 2:-2] #column 0 is patient_id and column 1 is time step, last col is mort label, 2nd to last is los
        X_test = test[0][:, :, 2:-2]
        X_train = X_train[:,:,7:]
        X_test = X_test[:,:,7:]

    elif not numerical and categorical:
        X_train = train[0][:, :, 2:-2] #column 0 is patient_id and column 1 is time step, last col is mort label, 2nd to last is los
        X_test = test[0][:, :, 2:-2]
        X_train = X_train[:,:,0:7]
        X_test = X_test[:,:,0:7]

    Y_train = train[0][:, 0, -1]
    Y_test = test[0][:, 0, -1]
   
    

    X_train = list(zip(X_train, nrows_train))
    
    if val:
        (X_train, Y_train), (X_val, Y_val) = train_test_split(X_train, Y_train, split_size=0.2)        
        X_val, nrows_val = zip(*X_val)
    
    X_train, nrows_train = zip(*X_train)

    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)
    X_train = np.array(X_train)
    if numerical and categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=True,categorical=True, batch_size=batch_size, train=True,phen=False)
        train_steps = np.ceil(len(X_train)/batch_size)
    if numerical and not categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=True,categorical=False, batch_size=batch_size, train=True,phen=False)
        train_steps = np.ceil(len(X_train)/batch_size)
    if not numerical and categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=False,categorical=True, batch_size=batch_size, train=True,phen=False)
        train_steps = np.ceil(len(X_train)/batch_size)
    if val:
        Y_val = Y_val.astype(int)        
        val_gen   = batch_generator(X_val, Y_val, batch_size=batch_size, train=False,phen=False)
        val_steps = np.ceil(len(X_val)/batch_size)

    max_time_step = nrows_test
    if val:
        return train_gen, train_steps, val_gen, val_steps, (X_test, Y_test), max_time_step

    return  train_gen, train_steps, (X_test, Y_test), max_time_step




def data_reader_for_model_rlos(config,train, test, numerical=config.num, categorical=config.cat, batch_size=1024, val=False):

    nrows_train = train[1]
    nrows_test = test[1]

    
    if numerical and categorical:
        X_train = train[0][:, :, 2:-2] #column 0 is patient_id and column 1 is time step
        X_test = test[0][:, :, 2:-2]

    elif numerical and not categorical:
        X_train = train[0][:, :, 2:-2] #column 0 is patient_id and column 1 is time step
        X_test = test[0][:, :, 2:-2]
        X_train = X_train[:,:,7:]
        X_test = X_test[:,:,7:]

    elif not numerical and categorical:
        X_train = train[0][:, :, 2:-2] #column 0 is patient_id and column 1 is time step
        X_test = test[0][:, :, 2:-2]
        X_train = X_train[:,:,0:7]
        X_test = X_test[:,:,0:7]
    
    Y_train = train[0][:, :, -1]
    Y_test = test[0][:, :, -1]

    X_train = list(zip(X_train, nrows_train))
    if val:
        (X_train, Y_train), (X_val, Y_val) = train_test_split(X_train, Y_train, split_size=0.2)        
        X_val, nrows_val = zip(*X_val)
    
    X_train, nrows_train = zip(*X_train)
    X_train = np.array(X_train)
    
    if numerical and categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=True,categorical=True, batch_size=batch_size, train=True,phen=False)
        train_steps = np.ceil(len(X_train)/batch_size)
    if numerical and not categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=True,categorical=False, batch_size=batch_size, train=True,phen=False)
        train_steps = np.ceil(len(X_train)/batch_size)
    if not numerical and categorical:
        train_gen = batch_generator(config,X_train, Y_train,numerical=False,categorical=True, batch_size=batch_size, train=True,phen=False)
        train_steps = np.ceil(len(X_train)/batch_size)

    if val:
        Y_val = Y_val.astype(int)        
        val_gen   = batch_generator(config,X_val, Y_val,numerical=True,categorical=True, batch_size=batch_size, train=False,phen=False)
        val_steps = np.ceil(len(X_val)/batch_size)

    max_time_step = nrows_test
    if val:
        return train_gen, train_steps, val_gen, val_steps, (X_test, Y_test), max_time_step

    return  train_gen, train_steps, (X_test, Y_test), max_time_step
