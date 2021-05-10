import numpy as np


########################################################################################
#                            Alias Sampling Implementations                            #
########################################################################################
def alias_init(prob, normalized = False):
    '''
    Alias sampling initialization.

    Parameters
    ----------
    prob: list, the probabilities of the events.
    normalized: whether the probabilities have been normalized.

    Returns
    -------
    The initialized parameters needed for alias sampling.

    References
    ----------
    Li, Aaron Q., et al. "Reducing the sampling complexity of topic models.", 
      Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.
    '''
    prob = np.array(prob, dtype = np.float)
    n_events = prob.shape[0]
    prob_norm = np.sum(prob) / n_events
    prob = prob / prob_norm

    small, large = [], []
    for i in range(n_events):
        p = prob[i]
        if p < 1.0:
            small.append(i)
        else:
            large.append(i)
    
    accept = np.zeros(n_events, dtype = np.float)
    alias = np.zeros(n_events, dtype = np.int)

    while small and large:
        small_idx = small.pop()
        large_idx = large.pop()

        accept[small_idx] = prob[small_idx]
        alias[small_idx] = large_idx
        prob[large_idx] = prob[large_idx] - (1 - prob[small_idx])

        if prob[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)
        
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1
    
    while large:
        large_idx = large.pop()
        accept[large_idx] = 1

    return (accept, alias)


def alias_sampling(params):
    '''
    Alias sampling method.

    Parameters
    ----------
    params: tuple of (accept, alias), the initialized parameters of alias sampling, which is given by "alias_init(...)"

    Returns
    -------
    The index of the sampled event.

    References
    ----------
    Li, Aaron Q., et al. "Reducing the sampling complexity of topic models.", 
      Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.
    '''
    (accept, alias) = params
    n_events = accept.shape[0]
    idx = int(np.floor(np.random.rand() * n_events))
    if np.random.rand() < accept[idx]:
        return idx
    else:
        return alias[idx]
