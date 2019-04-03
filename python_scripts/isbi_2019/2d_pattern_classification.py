# -*- coding: utf-8 -*-

"""
Localization pattern classification of RNA molecules in 2-d.
"""

import os
import argparse
import time

import bigfish.stack as stack
import bigfish.classification as classification

from .utils import encode_labels

# TODO build tensorflow from source to avoid the next line
# Your CPU supports instructions that this TensorFlow binary was not compiled
# to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if __name__ == '__main__':
    print()
    print("Running {0} file...". format(os.path.basename(__file__)), "\n")
    start_time = time.time()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input",
                        help="Path of the input data.",
                        type=str)
    parser.add_argument("log_directory",
                        help="Path of the log directory.",
                        type=str)
    parser.add_argument("--features",
                        help="Features used ('normal', 'distance' or "
                             "'surface').",
                        type=str,
                        default="normal")
    parser.add_argument("--classes",
                        help="Set of classes to predict.",
                        type=str,
                        default="all")
    parser.add_argument("--batch_size",
                        help="Size of a batch.",
                        type=int,
                        default=16)
    parser.add_argument("--nb_epochs",
                        help="Number of epochs to train the model.",
                        type=int,
                        default=10)
    parser.add_argument("--nb_workers",
                        help="Number of workers to use.",
                        type=int,
                        default=1)
    parser.add_argument("--multiprocessing",
                        help="Use multiprocessing.",
                        type=bool,
                        default=False)
    args = parser.parse_args()

    # parameters
    input_shape = (224, 224)

    print("------------------------")
    print("Input data: {0}".format(args.path_input))
    print("Output logs: {0}".format(args.log_directory), "\n")

    print("------------------------")
    print("Input shape: {0}".format(input_shape))
    print("Features: {0}".format(args.features))
    print("Batch size: {0}".format(args.batch_size))
    print("Number of epochs: {0}".format(args.nb_epochs))
    print("Number of workers: {0}".format(args.nb_workers))
    print("Multiprocessing: {0}".format(args.multiprocessing), "\n")

    print("--- PREPROCESSING ---", "\n")

    # load data
    df = stack.read_pickle(args.path_input)
    print("Shape input dataframe (before preparation): {0}".format(df.shape))

    # prepare data
    df, encoder, classes = encode_labels(df,
                                         column_name="pattern_name",
                                         classes_to_analyse="all")
    nb_classes = len(classes)
    print("Number of classes: {0}".format(nb_classes))
    print("Classes: {0}".format(classes))
    print("Shape input dataframe (after preparation): {0}".format(df.shape))
    print()

    # split data
    df_train, df_validation, df_test = stack.split_from_background(
        data=df,
        p_validation=0.2,
        p_test=0.2,
        logdir=args.log_directory)
    print("Split train|validation|test: {0}|{1}|{2}"
          .format(df_train.shape[0], df_validation.shape[0], df_test.shape[0]))

    # build train generator
    train_generator = stack.Generator(
        data=df_train,
        method=args.features,
        batch_size=args.batch_size,
        input_shape=input_shape,
        augmentation=True,
        with_label=True,
        nb_classes=nb_classes,
        nb_epoch_max=None)
    print("Number of train batches per epoch: {0}"
          .format(train_generator.nb_batch_per_epoch))

    # build validation generator
    validation_generator = stack.Generator(
        data=df_validation,
        method=args.features,
        batch_size=args.batch_size,
        input_shape=input_shape,
        augmentation=False,
        with_label=True,
        nb_classes=nb_classes,
        nb_epoch_max=None)
    print("Number of validation batches per epoch: {0}"
          .format(validation_generator.nb_batch_per_epoch))

    # build test generator
    test_generator = stack.Generator(
        data=df_test,
        method=args.features,
        batch_size=args.batch_size,
        input_shape=input_shape,
        augmentation=False,
        with_label=True,
        nb_classes=nb_classes,
        nb_epoch_max=None,
        shuffle=False)
    print("Number of test batches per epoch: {0}"
          .format(test_generator.nb_batch_per_epoch))
    print()

    print("--- TRAINING ---", "\n")

    # build and fit model
    model = classification.SqueezeNet0(
        nb_classes=nb_classes,
        bypass=True,
        optimizer="adam",
        logdir=args.log_directory)
    print("Model trained: {0}".format(model.trained))
    model.print_model()
    model.fit_generator(train_generator, validation_generator, args.nb_epochs,
                        args.nb_workers, args.multiprocessing)
    model.save_training_history()
    print("Model trained: {0}".format(model.trained))
    print()

    print("--- EVALUATION ---", "\n")

    # evaluate model with train data
    train_generator.reset()
    loss, accuracy = model.evaluate_generator(train_generator,
                                              args.nb_workers,
                                              args.multiprocessing,
                                              verbose=0)
    print("Loss train: {0:.3f} | Accuracy train: {1:.3f}"
          .format(loss, 100 * accuracy))

    # evaluate model with validation data
    validation_generator.reset()
    loss, accuracy = model.evaluate_generator(validation_generator,
                                              args.nb_workers,
                                              args.multiprocessing,
                                              verbose=0)
    print("Loss validation: {0:.3f} | Accuracy validation: {1:.3f}"
          .format(loss, 100 * accuracy))

    # evaluate model with test data
    loss, accuracy = model.evaluate_generator(test_generator,
                                              args.nb_workers,
                                              args.multiprocessing,
                                              verbose=0)
    print("Loss test: {0:.3f} | Accuracy test: {1:.3f}"
          .format(loss, 100 * accuracy))

    end_time = time.time()
    duration = int(round((end_time - start_time) / 60))
    print("Duration: {0} minutes.".format(duration))
