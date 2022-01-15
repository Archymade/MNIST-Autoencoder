import tensorflow as tf

def compute_loss(model, x, kl_factor = 1e-3):
    
    ''' Compute autoencoder loss. '''
    
    outputs = model.encode(x)
    
    if model.stochastic:
        outputs = model.reparametrize(outputs)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(0.5 * model.var_logvar) - 1 - model.var_logvar + tf.square(model.var_mean),
                                      axis = range(len(model.var_logvar.shape)))
        
    outputs = model.decode(outputs)
    
    rc_loss = tf.reduce_mean(tf.losses.mean_squared_error(x, outputs), axis = range(len(x.shape)))
    
    if model.stochastic:
        total_loss = rc_loss + (kl_factor * kl_loss)
    else:
        total_loss = rc_loss
    
    return total_loss