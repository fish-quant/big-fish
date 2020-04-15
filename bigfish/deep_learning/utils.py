from optparse import OptionParser
from skimage.measure import label
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import jaccard_similarity_score, f1_score
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from skimage.morphology import erosion, disk
from os.path import join
import os
from skimage.io import imsave, imread
import numpy as np
import pdb
import time
from progressbar import ProgressBar
from postproc.postprocessing import PostProcess, generate_wsl


def GetOptions():
    """
    Defines most of the options needed
    """
    parser = OptionParser()
    parser.add_option("--tf_record", dest="TFRecord", type="string",
                      default="",
                      help="Where to find the TFrecord file")
    parser.add_option("--path", dest="path", type="string",
                      help="Where to collect the patches")
    parser.add_option("--size_train", dest="size_train", type="int",
                      help="size of the input image to the network")
    parser.add_option("--log", dest="log",
                      help="log dir")
    parser.add_option("--learning_rate", dest="lr", type="float", default=0.01,
                      help="learning_rate")
    parser.add_option("--batch_size", dest="bs", type="int", default=1,
                      help="batch size")
    parser.add_option("--epoch", dest="epoch", type="int", default=1,
                      help="number of epochs")
    parser.add_option("--n_features", dest="n_features", type="int",
                      help="number of channels on first layers")
    parser.add_option("--weight_decay", dest="weight_decay", type="float",
                      default=0.00005,
                      help="weight decay value")
    parser.add_option("--dropout", dest="dropout", type="float",
                      default=0.5,
                      help="dropout value to apply to the FC layers.")
    parser.add_option("--mean_file", dest="mean_file", type="str",
                      help="where to find the mean file to substract to the original image.")
    parser.add_option('--n_threads', dest="THREADS", type=int, default=100,
                      help="number of threads to use for the preprocessing.")
    parser.add_option('--crop', dest="crop", type=int, default=4,
                      help="crop size depending on validation/test/train phase.")
    parser.add_option('--split', dest="split", type="str",
                      help="validation/test/train phase.")
    parser.add_option('--p1', dest="p1", type="int",
                      help="1st input for post processing.")
    parser.add_option('--p2', dest="p2", type="float",
                      help="2nd input for post processing.")
    parser.add_option('--iters', dest="iters", type="int")
    parser.add_option('--seed', dest="seed", type="int")
    parser.add_option('--size_test', dest="size_test", type="int")
    parser.add_option('--restore', dest="restore", type="str")
    parser.add_option('--save_path', dest="save_path", type="str", default=".")
    parser.add_option('--type', dest="type", type="str",
                      help="Type for the datagen")
    parser.add_option('--UNet', dest='UNet', action='store_true')
    parser.add_option('--no-UNet', dest='UNet', action='store_false')
    parser.add_option('--output', dest="output", type="str")
    parser.add_option('--output_csv', dest="output_csv", type="str")

    (options, args) = parser.parse_args()

    return options


def ComputeMetrics(prob, batch_labels, p1, p2, rgb=None, save_path=None,
                   ind=0):
    """
    Computes all metrics between probability map and corresponding label.
    If you give also an rgb image it will save many extra meta data image.
    """
    GT = label(batch_labels.copy())
    PRED = PostProcess(prob, p1, p2)
    # PRED = label((prob > 0.5).astype('uint8'))
    lbl = GT.copy()
    pred = PRED.copy()
    aji = AJI_fast(lbl, pred)
    lbl[lbl > 0] = 1
    pred[pred > 0] = 1
    l, p = lbl.flatten(), pred.flatten()
    acc = accuracy_score(l, p)
    roc = roc_auc_score(l, p)
    jac = jaccard_similarity_score(l, p)
    f1 = f1_score(l, p)
    recall = recall_score(l, p)
    precision = precision_score(l, p)
    if rgb is not None:
        xval_n = join(save_path, "xval_{}.png").format(ind)
        yval_n = join(save_path, "yval_{}.png").format(ind)
        prob_n = join(save_path, "prob_{}.png").format(ind)
        pred_n = join(save_path, "pred_{}.png").format(ind)
        c_gt_n = join(save_path, "C_gt_{}.png").format(ind)
        c_pr_n = join(save_path, "C_pr_{}.png").format(ind)

        imsave(xval_n, rgb)
        imsave(yval_n, color_bin(GT))
        imsave(prob_n, prob)
        imsave(pred_n, color_bin(PRED))
        imsave(c_gt_n, add_contours(rgb, GT))
        imsave(c_pr_n, add_contours(rgb, PRED))

    return acc, roc, jac, recall, precision, f1, aji


def color_bin(bin_labl):
    """
    Colors bin image so that nuclei come out nicer.
    """
    dim = bin_labl.shape
    x, y = dim[0], dim[1]
    res = np.zeros(shape=(x, y, 3))
    for i in range(1, bin_labl.max() + 1):
        rgb = np.random.normal(loc=125, scale=100, size=3)
        rgb[rgb < 0] = 0
        rgb[rgb > 255] = 255
        rgb = rgb.astype(np.uint8)
        res[bin_labl == i] = rgb
    return res.astype(np.uint8)


def add_contours(rgb_image, contour, ds=2):
    """
    Adds contours to images.
    The image has to be a binary image
    """
    rgb = rgb_image.copy()
    contour[contour > 0] = 1
    boundery = contour - erosion(contour, disk(ds))
    rgb[boundery > 0] = np.array([0, 0, 0])
    return rgb


def CheckOrCreate(path):
    """
    If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def Intersection(A, B):
    """
    Returns the pixel count corresponding to the intersection
    between A and B.
    """
    C = A + B
    C[C != 2] = 0
    C[C == 2] = 1
    return C


def Union(A, B):
    """
    Returns the pixel count corresponding to the union
    between A and B.
    """
    C = A + B
    C[C > 0] = 1
    return C


def AssociatedCell(G_i, S):
    """
    Returns the indice of the associated prediction cell for a certain
    ground truth element. Maybe do something if no associated cell in the
    prediction mask touches the GT.
    """

    def g(indice):
        S_indice = np.zeros_like(S)
        S_indice[S == indice] = 1
        NUM = float(Intersection(G_i, S_indice).sum())
        DEN = float(Union(G_i, S_indice).sum())
        return NUM / DEN

    res = map(g, range(1, S.max() + 1))
    indice = np.array(res).argmax() + 1
    return indice


pbar = ProgressBar()


def AJI(G, S):
    """
    AJI as described in the paper, AJI is more abstract implementation but 100times faster.
    """
    G = label(G, background=0)
    S = label(S, background=0)

    C = 0
    U = 0
    USED = np.zeros(S.max())

    for i in pbar(range(1, G.max() + 1)):
        only_ground_truth = np.zeros_like(G)
        only_ground_truth[G == i] = 1
        j = AssociatedCell(only_ground_truth, S)
        only_prediction = np.zeros_like(S)
        only_prediction[S == j] = 1
        C += Intersection(only_prediction, only_ground_truth).sum()
        U += Union(only_prediction, only_ground_truth).sum()
        USED[j - 1] = 1

    def h(indice):
        if USED[indice - 1] == 1:
            return 0
        else:
            only_prediction = np.zeros_like(S)
            only_prediction[S == indice] = 1
        return only_prediction.sum()

    U_sum = map(h, range(1, S.max() + 1))
    U += np.sum(U_sum)
    return float(C) / float(U)


def AJI_fast(G, S):
    """
    AJI as described in the paper, but a much faster implementation.
    """
    G = label(G, background=0)
    S = label(S, background=0)
    if S.sum() == 0:
        return 0.
    C = 0
    U = 0
    USED = np.zeros(S.max())

    G_flat = G.flatten()
    S_flat = S.flatten()
    G_max = np.max(G_flat)
    S_max = np.max(S_flat)
    m_labels = max(G_max, S_max) + 1
    cm = confusion_matrix(G_flat, S_flat, labels=range(m_labels)).astype(
        np.float)
    LIGNE_J = np.zeros(S_max)
    for j in range(1, S_max + 1):
        LIGNE_J[j - 1] = cm[:, j].sum()

    for i in range(1, G_max + 1):
        LIGNE_I_sum = cm[i, :].sum()

        def h(indice):
            LIGNE_J_sum = LIGNE_J[indice - 1]
            inter = cm[i, indice]

            union = LIGNE_I_sum + LIGNE_J_sum - inter
            return inter / union

        JI_ligne = map(h, range(1, S_max + 1))
        best_indice = np.argmax(JI_ligne) + 1
        C += cm[i, best_indice]
        U += LIGNE_J[best_indice - 1] + LIGNE_I_sum - cm[i, best_indice]
        USED[best_indice - 1] = 1

    U_sum = ((1 - USED) * LIGNE_J).sum()
    U += U_sum
    return float(C) / float(U)