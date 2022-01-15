### Ensuring reproducibility

import os
import random
import tensorflow as tf
import numpy as np

def Reproduce(inter_threads = 3, intra_threads = 1, seed = 2021):
    ### OS reproducibility
    os.environ['PYTHONHASHSEED'] = '0'
    
    ### NumPy reproducibility
    np.random.default_rng(seed)
    random.seed(seed)

    ### Tensorflow reproducibility
    tf.random.set_seed(seed)

    ### Session configuration
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    
    return None

### Create computational graph via session config
#ssess = tf.Session(graph = tf.get_default_graph(), config = sess_config)

### Set the graph above as default graph
#K.set_session(sess)