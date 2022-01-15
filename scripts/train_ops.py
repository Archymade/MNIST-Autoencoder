import tensorflow as tf
from losses import compute_loss
from itertools import chain

@tf.function
def train_step(model, x, kl_factor):
    
    with tf.GradientTape() as model_tape:
        loss = compute_loss(model, x, kl_factor)
    
    gradients = model_tape.gradient(loss, chain(model.encoder.trainable_variables,
                                                model.encoder.trainable_variables))
    model.optimizer.apply_gradients(zip(gradients, chain(model.encoder.trainable_variables,
                                                         model.encoder.trainable_variables)))
    
    return None