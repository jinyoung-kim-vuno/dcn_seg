from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam


def select(optimizer, initial_lr):

    if optimizer == 'SGD':
        return SGD(lr=initial_lr, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'RMSprop':
        return RMSprop(lr=initial_lr, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer == 'Adagrad':
        return Adagrad(lr=initial_lr, epsilon=None, decay=0.0)
    elif optimizer == 'Adadelta':
        return Adadelta(lr=initial_lr, rho=0.95, epsilon=None, decay=0.0)
    elif optimizer == 'Adam':
        return Adam(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif optimizer == 'Adamax':
        return Adamax(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    elif optimizer == 'Nadam':
        return Nadam(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
