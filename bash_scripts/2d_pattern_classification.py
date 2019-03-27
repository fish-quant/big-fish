# -*- coding: utf-8 -*-

"""
Localization pattern classification of RNA molecules in 2-d.
"""

import os
import argparse
import pickle

import bigfish.stack as stack
import bigfish.classification as classification

# TODO build tensorflow from source to avoid the next line
# Your CPU supports instructions that this TensorFlow binary was not compiled
# to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    print()
    print("Running {0} file...". format(os.path.basename(__file__)), "\n")

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input",
                        help="Path of the input data.",
                        type=str)
    parser.add_argument("log_directory",
                        help="Path of the log directory.",
                        type=str)
    parser.add_argument("--batch_size",
                        help="Size of a batch.",
                        type=int,
                        default=16)
    parser.add_argument("--input_shape",
                        help="Shape of the input data.",
                        type=tuple,
                        default=(224, 224))
    parser.add_argument("--nb_classes",
                        help=" Final number of classes.",
                        type=int,
                        default=9)
    parser.add_argument("--nb_epochs",
                        help="Number of epochs to train the model.",
                        type=int,
                        default=10)
    args = parser.parse_args()

    print("------------------------")
    print("Input data: {0}".format(args.path_input))
    print("Output logs: {0}".format(args.log_directory), "\n")

    print("------------------------")
    print("Input shape: {0}".format(args.input_shape))
    print("Batch size: {0}".format(args.batch_size))
    print("Number of classes: {0}".format(args.nb_classes))
    print("Number of epochs: {0}".format(args.nb_epochs), "\n")

    print("--- PREPROCESSING ---", "\n")

    # load data
    # path_output = os.path.join(main_directory, "data_cleaned_small")
    with open(args.path_input, mode='rb') as f:
        df = pickle.load(f)
    print("Shape input dataframe (before preparation): {0}".format(df.shape))

    # prepare data
    classes = ["inNUC", "cell2D", "nuc2D", "foci", "polarized", "cellext",
               "random"]
    query = "pattern_name in {0}".format(str(classes))
    df = df.query(query)
    print("Shape input dataframe (after preparation): {0}".format(df.shape))
    df_train, df_validation, df_test = stack.split_from_background(
        data=df,
        p_validation=0.2,
        p_test=0.2)
    print("Split train|validation|test: {0}|{1}|{2}"
          .format(df_train.shape[0], df_validation.shape[0], df_test.shape[0]))

    # build train generator
    train_generator = stack.Generator(
        data=df_train,
        method="normal",
        batch_size=args.batch_size,
        input_shape=args.input_shape,
        augmentation=True,
        with_label=True,
        nb_classes=args.nb_classes,
        nb_epoch_max=None)
    print("Number of train batches per epoch: {0}"
          .format(train_generator.nb_batch_per_epoch))

    # build validation generator
    validation_generator = stack.Generator(
        data=df_validation,
        method="normal",
        batch_size=args.batch_size,
        input_shape=args.input_shape,
        augmentation=False,
        with_label=True,
        nb_classes=args.nb_classes,
        nb_epoch_max=1)
    print("Number of validation batches per epoch: {0}"
          .format(validation_generator.nb_batch_per_epoch))
    print()

    print("--- TRAINING ---", "\n")

    # build and fit model
    model = classification.SqueezeNet0(
        nb_classes=args.nb_classes,
        bypass=False,
        optimizer="adam",
        logdir=args.log_directory)
    print("Model trained: {0}".format(model.trained))
    model.print_model()
    model.fit_generator(train_generator, validation_generator, args.nb_epochs)
    print()

    print("--- EVALUATION ---", "\n")

    # evaluate model
    print("Model trained: {0}".format(model.trained))
    loss, accuracy = model.evaluate_generator(validation_generator)
    print("Loss: {0} | Accuracy: {1}".format(loss, 100 * accuracy))
