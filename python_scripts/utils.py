# -*- coding: utf-8 -*-

"""
Utility functions.
"""

from sklearn.preprocessing import LabelEncoder


def encode_labels(data, column_name="pattern_name", classes_to_analyse="all"):
    """Filter classes we want to analyze and encode them from a string format
    to a numerical one.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with a feature containing the label in string format.
    column_name : str
        Name of the feature to use in the dataframe as label.
    classes_to_analyse : str
        Define the set of classe we want to keep and to encode before training
        a model:
        - 'experimental' to fit with the experimental data (5 classes).
        - '2d' to analyze the 2-d classes only (7 classes).
        - 'all' to analyze all the classes (9 classes).

    Returns
    -------
    data : pd.DataFrame
        Dataframe with the encoded label in an additional column 'label'. If
        the original columns label is already named 'label', we rename both
        columns 'label_str' and 'label_num'.
    encoder : sklearn.preprocessing.LabelEncoder
        Fitted encoder to encode of decode a label.
    classes : List[str]
        List of the classes to keep and encode.

    """
    # experimental analysis
    if classes_to_analyse == "experimental":
        data, encoder, classes = _encode_label_experimental(data, column_name)
    # 2-d analysis
    elif classes_to_analyse == "2d":
        data, encoder, classes = _encode_label_2d(data, column_name)
    # complete analysis
    elif classes_to_analyse == "all":
        data, encoder, classes = _encode_label_all(data, column_name)
    else:
        raise ValueError("'classes_to_analyse' can only take three values: "
                         "'experimental', '2d' or 'all'.")

    return data, encoder, classes


def _encode_label_experimental(data, column_name):
    """Filter the 5 classes included in the experimental dataset, then encode
    them from a string format to a numerical one.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with a feature containing the label in string format.
    column_name : str
        Name of the feature to use in the dataframe as label.

    Returns
    -------
    data : pd.DataFrame
        Dataframe with the encoded label in an additional column 'label'. If
        the original columns label is already named 'label', we rename both
        columns 'label_str' and 'label_num'.
    encoder : sklearn.preprocessing.LabelEncoder
        Fitted encoder to encode of decode a label.
    classes : List[str]
        List of the classes to keep and encode.

    """
    # get classes to use
    classes = ["random", "foci", "cellext", "inNUC", "nuc2D"]

    # fit a label encoder
    encoder = LabelEncoder()
    encoder.fit(classes)

    # filter rows
    query = "{0} in {1}".format(column_name, str(classes))
    data = data.query(query)

    # encode labels
    if column_name == "label":
        data = data.assign(
            label_str=data.loc[:, column_name],
            label_num=encoder.transform(data.loc[:, column_name]))
    else:
        data = data.assign(
            label=encoder.transform(data.loc[:, column_name]))

    return data, encoder, classes


def _encode_label_2d(data, column_name):
    """Filter the 2-d classes, then encode them from a string format to a
    numerical one.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with a feature containing the label in string format.
    column_name : str
        Name of the feature to use in the dataframe as label.

    Returns
    -------
    data : pd.DataFrame
        Dataframe with the encoded label in an additional column 'label'. If
        the original columns label is already named 'label', we rename both
        columns 'label_str' and 'label_num'.
    encoder : sklearn.preprocessing.LabelEncoder
        Fitted encoder to encode of decode a label.
    classes : List[str]
        List of the classes to keep and encode.

    """
    # get classes to use
    classes = ["random", "foci", "cellext", "inNUC", "nuc2D", "cell2D",
               "polarized"]

    # fit a label encoder
    encoder = LabelEncoder()
    encoder.fit(classes)

    # filter rows
    query = "{0} in {1}".format(column_name, str(classes))
    data = data.query(query)

    # encode labels
    if column_name == "label":
        data = data.assign(
            label_str=data.loc[:, column_name],
            label_num=encoder.transform(data.loc[:, column_name]))
    else:
        data = data.assign(
            label=encoder.transform(data.loc[:, column_name]))

    return data, encoder, classes


def _encode_label_all(data, column_name):
    """Encode all the classes from a string format to a numerical one.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with a feature containing the label in string format.
    column_name : str
        Name of the feature to use in the dataframe as label.

    Returns
    -------
    data : pd.DataFrame
        Dataframe with the encoded label in an additional column 'label'. If
        the original columns label is already named 'label', we rename both
        columns 'label_str' and 'label_num'.
    encoder : sklearn.preprocessing.LabelEncoder
        Fitted encoder to encode of decode a label.
    classes : List[str]
        List of the classes to keep and encode.

    """
    # get classes to use
    classes = ["random", "foci", "cellext", "inNUC", "nuc2D", "cell2D",
               "polarized", "cell3D", "nuc3D"]

    # fit a label encoder
    encoder = LabelEncoder()
    encoder.fit(classes)

    # filter rows
    query = "{0} in {1}".format(column_name, str(classes))
    data = data.query(query)

    # encode labels
    if column_name == "label":
        data = data.assign(
            label_str=data.loc[:, column_name],
            label_num=encoder.transform(data.loc[:, column_name]))
    else:
        data = data.assign(
            label=encoder.transform(data.loc[:, column_name]))

    return data, encoder, classes
