from .treeinterpreter import _predict_tree
from sklearn.ensemble.forest import _generate_unsampled_indices, _generate_sample_indices
import numpy as np
def feature_importance(rf, X, y, type = 'oob'):
    n_samples, n_features = X.shape
    if len(y.shape) != 2:
        raise ValueError('y must be 2d array (n_samples, 1) if numerical or (n_samples, n_categories).')
    out = np.zeros((n_features,))
    for tree in rf.estimators_:
        if type == 'oob':
            unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
            _, _, contributions = _predict_tree(tree, X[unsampled_indices])
            out +=  np.tensordot(y[unsampled_indices,:], contributions, axes=([0, 1], [0, 2])) / len(unsampled_indices)
        elif type == 'test':
            _, _, contributions = _predict_tree(tree, X)
            out +=  np.tensordot(y, contributions, axes=([0, 1], [0, 2])) / n_samples
        elif type == 'classic':
            sampled_indices = _generate_sample_indices(tree.random_state, n_samples)
            _, _, contributions = _predict_tree(tree, X[sampled_indices])
            out +=  np.tensordot(y[sampled_indices,:], contributions, axes=([0, 1], [0, 2])) / len(sampled_indices)
        else:
            raise ValueError('type is not recognized. (%s)'%(type))
    out /= rf.n_estimators
    if np.sum(out[out > 0]) + 10 * np.sum(out[out < 0]) < 0:
        return out
    else:
        return out / np.sum(out) 
