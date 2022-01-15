#####################################################################################################################################

    if args.load_weights & len(os.listdir(args.D)) != 0:
        autoencoder.load_weights(args.D)
        print('Weights loaded successfully.')


    elif args.load_weights & len(os.listdir(args.D)) == 0:
        raise Exception(f'Weights directory {(args.D)} contains no saved weights. Model requires training.')
        exit()

    else:
        if args.train:
            train_x = load_dataset(DIR=args.data_dir, input_shape=args.input_shape[:2],
                                   batch_size=args.batch_size)

            if args.load_weights:
                print('Resuming training >>>')

            for epoch in range(args.epochs):
                for batch in train_x:
                    train_step(autoencoder, batch, args.klf)

                generate_and_save_images(autoencoder, sample[0], folder=args.results_dir,
                                         epoch=epoch, style=args.style, save_image=args.save,
                                         variational=args.variational)