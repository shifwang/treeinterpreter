from .treeinterpreter import _predict_tree
from sklearn.ensemble.forest import _generate_unsampled_indices, _generate_sample_indices
import numpy as np
def feature_importance(rf, X, y, type = 'oob', normalized = True):
    n_samples, n_features = X.shape
    if len(y.shape) != 2:
        raise ValueError('y must be 2d array (n_samples, 1) if numerical or (n_samples, n_categories).')
    out = np.zeros((n_features,))
    SE = np.zeros((n_features,))
    for tree in rf.estimators_:
        if type == 'oob':
            if rf.bootstrap:
                indices = _generate_unsampled_indices(tree.random_state, n_samples)
            else:
                raise ValueError('Without bootstrap, it is not possible to calculate oob.')
        elif type == 'test':
            indices = np.arange(n_samples)
        elif type == 'classic':
            if rf.bootstrap:
                indices = _generate_sample_indices(tree.random_state, n_samples)
            else:
                indices = np.arange(n_samples)
        else:
            raise ValueError('type is not recognized. (%s)'%(type))
        _, _, contributions = _predict_tree(tree, X[indices,:])
        if len(contributions.shape) == 2:
            contributions = contributions[:,:,np.newaxis]
        tmp =  np.tensordot(y[indices,:], contributions, axes=([0, 1], [0, 2])) 
        out +=  tmp / sum(tmp) if normalized else tmp
        SE += (tmp / sum(tmp)) ** 2 if normalized else tmp ** 2
    out /= rf.n_estimators
    SE /= rf.n_estimators
    SE = ((SE - out ** 2) / rf.n_estimators) ** .5 
    return out, SE
    #if np.sum(out[out > 0]) + 10 * np.sum(out[out < 0]) < 0:
    #    return out
    #else:
    #    return out / np.sum(out) 
