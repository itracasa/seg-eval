import rasterio
import geopandas as gpd
from shapely.geometry import shape, box
from rasterio.features import shapes as rio_shapes
from rasterio.features import rasterize as rio_rasterize
import numpy as np
from skimage.morphology import dilation, erosion
from scipy.spatial import cKDTree

_eps = 1e-6

def np_to_shp(x, transform=None, crs=None, simplify=True):
    """
    Converts a segmentation mask (x) into a GeoDataFrame (shp). Output shapes will be georreferenced if affine transformation
    (transform) and coordinate reference system (crs) are set.
    """
    transform = transform if transform is not None else rasterio.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    shp = gpd.GeoDataFrame(
        {'geometry': [shape(g) for g, v in rio_shapes(x, transform=transform, mask=x>0)]}, 
        crs=crs
    )
    if simplify:
        shp = gpd.GeoDataFrame({'geometry': [shp.buffer(0).unary_union]}, crs=crs) # Buffer 0 to avoid invalid geometries
    return shp

def shp_to_np(gdf, out_shape, out_transform, all_touched=False):
    """
    Converts a GeoDataFrame (gdf) into a numpy array given an output shape (out_shape) and the
    corresponding affine transformation (out_transform).
    """
    np_array = rio_rasterize(gdf.geometry, out_shape=out_shape, transform=out_transform, all_touched=all_touched)
    return np_array

def get_buffer_np(y_true, alpha, kind='symmetric'):
    """
    Computes a tolerance buffer given the distance (in pixels). Pixels within tolerance
    buffer should be ignored for metric relaxation.
    """
    buff_outer = np.array(y_true)
    buff_inner = np.array(y_true)

    for i in range(alpha): buff_outer = dilation(buff_outer)
    for i in range(alpha): buff_inner = erosion(buff_inner)

    buffer = np.logical_or((buff_outer - y_true).astype('bool'), (y_true - buff_inner).astype('bool'))
    
    if kind == 'outer': 
        return buff_outer
    elif kind == 'inner':
        return buff_inner
    else: # symmetric
        return buffer

def get_buffer_shp(y_true_shp, alpha, kind='symmetric'):
    """
    Computes a tolerance buffer given the distance (in pixels). Pixels within tolerance
    buffer should be ignored for metric relaxation. It must be noted that y_true_shp must
    be a GeoDataFrame. For optimal performance y_true_shp should contain only a Multipolygon
    compressing all the shapes within the segmentation map.
    """
    buff_outer = y_true_shp.buffer(alpha, cap_style=3).iloc[0].buffer(0) # Buffer 0 to avoid invalid geometries
    buff_inner = y_true_shp.buffer(-alpha, cap_style=3).iloc[0].buffer(0) # Buffer 0 to avoid invalid geometries
    
    buff_outer = buff_outer.difference(y_true_shp.iloc[0].geometry)
    buff_inner = y_true_shp.iloc[0].geometry.difference(buff_inner)    
    buffer = buff_outer.union(buff_inner)
    
    if kind == 'outer': 
        return buff_outer
    elif kind == 'inner':
        return buff_inner
    else: # symmetric
        return buffer

def compute_confusion_matrix_np(y_pred, y_true, buffer=None):
    """
    Computes True Positives (TP), True Negatives (TN), False Positives (FP) and False Negatives (FN)
    given the predicted (y_pred) and ground truth (y_true) binary segmentation maps. Moreover, metrics
    can be relaxed applying a tolerance buffer (buffer). 
    """
    y_true = y_true.astype('float')
    y_pred = y_pred.astype('float')
    
    if buffer:
        y_true = y_true[~buffer]
        y_pred = y_pred[~buffer]

    tp = (y_true * y_pred).sum()
    tn = ((1-y_true) * (1-y_pred)).sum()
    fp = ((1-y_true) * y_pred).sum()
    # fn = y_true.size - tp - tn - fp
    fn = (y_true * (1-y_pred)).sum()
    
    return tp, tn, fp, fn

def compute_confusion_matrix_shp(y_pred_shp, y_true_shp, buffer=None, boundary=None):
    """
    Computes True Positives (TP), True Negatives (TN), False Positives (FP) and False Negatives (FN)
    given the predicted (y_pred_shp) and ground truth (y_true_shp) binary segmentation maps as GeoDataFrames. 
    For better performance GeoDataFrames should contain only a Multipolygon compressing all the shapes within the 
    segmentation map. Moreover, metrics can be relaxed applying a tolerance buffer (buffer). 
    """
    if boundary is None:
        boundary = box(*y_true_shp.total_bounds)
        
    y_true_shp = y_true_shp.buffer(0)
    y_pred_shp = y_pred_shp.buffer(0)
    
    y_true_shp = y_true_shp.geometry.iloc[0] if all(['Polygon' in dtype for dtype in y_true_shp.type.unique()]) else y_true_shp
    y_pred_shp = y_pred_shp.geometry.iloc[0] if all(['Polygon' in dtype for dtype in y_pred_shp.type.unique()]) else y_pred_shp
       
    not_y_true_shp = boundary.difference(y_true_shp)
    not_y_pred_shp = boundary.difference(y_pred_shp)
    
    if buffer:
        y_true_shp = y_true_shp.difference(buffer)
        y_pred_shp = y_pred_shp.difference(buffer)
        not_y_true_shp = not_y_true_shp.difference(buffer)
        not_y_pred_shp = not_y_pred_shp.difference(buffer)

    tp = y_true_shp.intersection(y_pred_shp).area
    tn = not_y_true_shp.intersection(not_y_pred_shp).area
    fp = not_y_true_shp.intersection(y_pred_shp).area
    # fn = boundary.area - tp - tn - fp
    fn = y_true_shp.intersection(not_y_pred_shp).area
    
    return tp, tn, fp, fn

def iou(tp, tn, fp, fn):
    """ 
    Jaccard Index, Intersection over Union
    """
    if tp + tn + fp + fn == tp + tn: return 1.
    iou = (tp + _eps) / (tp + fp + fn + _eps)
    return iou

def dice(tp, tn, fp, fn):
    """
    F1-score, F-score, Sørensen–Dice Coefficient, Dice Coefficient
    """
    if tp + tn + fp + fn == tp + tn: return 1.
    dice = (2 * tp + _eps) / (2 * tp + fp + fn + _eps)
    return dice

def ppv(tp, tn, fp, fn):
    """
    Precision, Positive Predicted Value
    """
    if tp + tn + fp + fn == tp + tn: return 1.
    ppv = (tp + _eps) / (tp + fp + _eps)
    return ppv

def tpr(tp, tn, fp, fn):
    """
    Sensitivity, Recall, True Positive Rate, Overall Accuracy, Detection Probability, Hit Rate
    """
    if tp + tn + fp + fn == tp + tn: return 1.
    tpr = (tp + _eps) / (tp + fn + _eps)
    return tpr

def tnr(tp, tn, fp, fn):
    """
    Specificity, True Negative Rate
    """
    if tp + tn + fp + fn == tp + tn: return 1.
    tnr = (tn + _eps) / (tn + fp + _eps)
    return tnr

def auc(tp, tn, fp, fn):
    """
    Area Under the Curve
    """
    if tp + tn + fp + fn == tp + tn: return 1. # Predition completely overlays ground 
    auc = 1 - 0.5 * (((fp + _eps) / (fp + tn + _eps)) + ((fn + _eps) / (fn + tp + _eps)))
    return auc

def kappa(tp, tn, fp, fn):
    """
    Cohen's Kappa
    """
    if tp + tn + fp + fn == tp + tn: return 1.
    fc = ((tn + fn) * (tn + fp) + (fp + tp) * (fn + tp) + _eps) / (tp + tn + fn + fp + _eps)
    kappa = (tp + tn - fc + _eps) / (tp + tn + fn + fp - fc + _eps)
    return kappa

def mcc(tp, tn, fp, fn):
    """
    Matthews correlation coefficient
    """
    if tp + tn + fp + fn == tp + tn: return 1. # Predition completely overlays ground truth
    mcc = (tp * tn - fp * fn + _eps) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + _eps) ** 0.5
    return mcc

def fnr(tp, tn, fp, fn):
    """
    False Negative Rate, Miss Rate
    """
    if tp + tn + fp + fn == tp + tn: return 0.
    fnr = (fn + _eps) / (fn + tp + _eps)
    return fnr

def fpr(tp, tn, fp, fn):
    """
    False Positive Rate, Fall-out
    """
    if tp + tn + fp + fn == tp + tn: return 0.
    fpr = (fp + _eps) / (fp + tn + _eps)
    return fpr

def hd(y_pred, y_true, method='modified', buffer=None):
    """
    Calculate the Hausdorff distance between nonzero elements of given segmentation maps. There are two methods
    available; standard [1] and modified [2]. The Hausdorff distance can be relaxed applying a tolerance buffer 
    (buffer) which is a boolean numpy array specifying which pixels should be ignored. 
    
    From https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py
    
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    .. [2] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
       matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
       :DOI:`10.1109/ICPR.1994.576361`
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.8155
    """
    y_true = y_true.astype('float')
    y_pred = y_pred.astype('float')
    
    if isinstance(buffer, np.ndarray):
        y_true = y_true[~buffer]
        y_pred = y_pred[~buffer]
        
    a_points = np.transpose(np.nonzero(y_pred[0].astype(np.bool)))
    b_points = np.transpose(np.nonzero(y_true[0].astype(np.bool)))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    fwd, bwd = (
        cKDTree(a_points).query(b_points, k=1)[0],
        cKDTree(b_points).query(a_points, k=1)[0],
    )
    
    if method not in ('standard', 'modified'):
        raise ValueError(f'unrecognized method {method}')

    if method == 'standard':  # standard Hausdorff distance
        return max(max(fwd), max(bwd))
    elif method == 'modified':  # modified Hausdorff distance
        return max(np.mean(fwd), np.mean(bwd))  
    
# Function synonyms
jaccard = iou
fscore = f1score = dice
precision = ppv
sensitivity = recall = oa = tpr
specificity = tnr
hausdorff = hausdorff_distance = hd