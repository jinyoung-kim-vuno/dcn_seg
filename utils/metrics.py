
from keras import backend as K
from utils.mathutils import measure_msd

# def rmse(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
#
def acc_dc(y_true, y_pred):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice_coef


def select(metric_opt):

    # metric = None
    # if metric_opt == 'accuracy':
    #     metric = ['acc']
    # elif metric_opt == 'Dice':
    #     metric = [dice]
    # elif metric_opt == 'all':
    #     metric = ['acc', dice]
    # else:
    #     print('Unknown metric')
    #print('metric for early stopping: val_' + metric_opt)
    return ['acc', acc_dc]


