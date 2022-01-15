import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers

from data_ops import load_dataset
from tfreproduce import Reproduce
from autoencoder import AutoEncoder

### Ensure reproducibility
Reproduce(intra_threads=1, inter_threads=3)


def configure_args():
    parser = argparse.ArgumentParser(description="Provide relevant CLI arguments.")

    parser.add_argument('--epochs', type=int, default=20, help='Number of training rounds')

    parser.add_argument('--variational', '-v', type=bool, default=True, choices=[True, False],
                        help='Variational autoencoder or deterministic autoencoder')

    parser.add_argument('--train', type=bool, default=True, choices=[True, False], help='Train model')
    
    parser.add_argument('--resume', '-r', type=bool, default = False, choices = [True, False],
                        help='Resume training from Checkpoint')

    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD', 'RMSprop'],
                        help='Optimizer algorithm for autoencoder training')

    parser.add_argument('--input_shape', default=[28, 28, 1], type=list,
                        help='Expected input image shape: [height, width, channels]')

    parser.add_argument('--latent_dims', type=int, default=100,
                        help='Dimensionality of latent vector space')

    parser.add_argument('--style', default='gruvboxd', type=str,
                        choices=['gruvboxd', 'solarizedd', 'solarizedl', 'onedork', 'oceans16', 'normal'],
                        help='Image visualization style')

    parser.add_argument('--save', default=True, choices=[True, False], type=bool,
                        help='Save sample images')

    parser.add_argument('--load_weights', default=True, type=bool, choices=[True, False],
                        help='Load pretrained weights')

    parser.add_argument('--log-dir', '-D', default=os.path.join(os.getcwd().replace('scripts', ''), 'artefacts'), type=str,
                        help='Storage directory for weights')

    parser.add_argument('--batch_size', default=32, type=int, help='Size for data batching')

    parser.add_argument('--data_dir', type=str, help='Local location of dataset',
                        default=os.path.join(os.getcwd().replace('scripts', ''), 'trainingSet', 'trainingSet'))

    parser.add_argument('--results_dir', type=str, default=os.path.join(os.getcwd().replace('scripts', ''), 'results'),
                        help='Local location to store generated images')

    parser.add_argument('--lr', default=3e-4, type=float, help='Convergence rate')

    parser.add_argument('--beta1', default=0.5, type=float, help='First moment')

    parser.add_argument('--beta2', default=0.999, type=float, help='Second moment')
    
    parser.add_argument('--klf', type=int, default = 10, help='KL-Loss factor')

    return parser


def check_args(args):
    ''' Validate CLI arguments. '''

    ### Epochs
    assert (args.epochs >= 1) & (type(args.epochs) == int), 'Epochs must be an object of type Int not less than 1.'

    ### Betas
    assert (type(args.beta1) == float) & (0 <= args.beta1 <= 1), 'Beta 1 must be in [0, 1).'

    assert (type(args.beta2) == float) & (0 <= args.beta2 <= 1), 'Beta 2 must be in [0, 1).'

    ### Learning rate
    assert (type(args.lr) == float), 'Learning rate must be in of type Float.'

    ### Input shape
    assert len(args.input_shape) == 3, 'Input shape must contain: [height, width, channels].'

    assert len(
        [*filter(lambda x: type(x) != int, args.input_shape)]) == 0, 'Input shape must comprise objects of type Int.'

    ### Latent space
    assert (type(args.latent_dims) == int) & (
                args.latent_dims >= 1), 'Latent size must be of type Int, and greater than or equal 1.'

    ### Batch size
    assert (type(args.batch_size) == int) & (args.batch_size >= 1), 'Batch size must be of type Int, and greater than or equal 1.'

    assert not (args.batch_size % 2), 'Batch size must be multiple of 2.'

    if not args.train:
        assert args.load_weights == True, 'Weights must be loaded if model is not to be trained.'

    ### If all's well
    return args


def main():
    ### Instantiate and validate arguments
    args = configure_args().parse_args()

    if args is None:
        exit()

    else:
        args = check_args(args)

    print(args)

    ### Instantiate optimizer object
    if args.optimizer == 'Adam':
        optimizer = optimizers.Adam(learning_rate=args.lr, beta_1=args.beta1, beta_2=args.beta2)

    elif args.optimizer == 'RMSprop':
        optimizer = optimizers.RMSprop(learning_rate=args.lr)

    else:
        optimizer = optimizers.SGD(learning_rate=args.lr)

    ### Instantiate model
    print('>>> Instantiating autoencoder object...')
    autoencoder = AutoEncoder(optimizer=optimizer, in_dims=args.input_shape,
                              latent_dims=args.latent_dims, variational=args.variational)
    print('>>> Done!', end = '\n')

    ### If we wish to train
    if args.train:

        ### Load weights first?
        if args.load_weights:

            if len(os.listdir(args.D)) != 0:
                autoencoder.load_weights(args.D)
                print('Resuming training >>>')

            else:
                raise Exception(f'Weights directory ({args.D}) contains no saved weights. Model requires training.')
                exit()

        ### Train from scratch?
        else:
            print('Starting training >>>')

        train_x = load_dataset(DIR=args.data_dir, input_shape=args.input_shape[:2],
                               batch_size=args.batch_size)

        sample = next(iter(train_x.skip(1)))
        sample = sample[0]

        ### Actual training
        for epoch in range(args.epochs):
            for batch in train_x:
                train_step(autoencoder, batch, args.klf)

            generate_and_save_images(autoencoder, sample=sample, folder=args.results_dir,
                                     epoch=epoch, style=args.style, save_image=args.save,
                                     variational=args.variational)
    ### If we do not wish to train
    else:
        if args.load_weights:
            autoencoder.load_weights(os.path.join(args.D, f'{input('Input checkpoint name: ')}.h5'))
            print('Model weights loaded.\nWaiting for next action...')


### Run program
if __name__ == '__main__':
    main()
