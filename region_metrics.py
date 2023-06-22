from shapely.geometry import Point, LineString, LinearRing, MultiPolygon, Polygon
import numpy as np
from fastprogress import progress_bar
import geopandas as gpd
import pandas as pd
from metrics import *
from shapely.ops import voronoi_diagram as svd

def LinearRingDense(linear_ring, step=1):
    xy=[]
    for i_vertex, (x, y) in enumerate(linear_ring.coords[:-1]):
        xy.append((x,y))
        segment = LineString([(x,y),linear_ring.coords[i_vertex+1]])
        n_additional_vertex = int(segment.length/step)
        segment_step = segment.length/(n_additional_vertex+1) # +1
        if n_additional_vertex>0:
            for delta in np.arange(0, segment.length, segment_step): 
                new_point = segment.interpolate(delta)
                xy.append((new_point.x,new_point.y))
    return LinearRing(xy)

def PolygonDense(polygon, step):
    exterior_dense = LinearRingDense(polygon.exterior, step)
    holes_dense =[LinearRingDense(interior, step) for interior in polygon.interiors]
    return Polygon(exterior_dense, holes=holes_dense)

def MultiPolygonDense(multi_polygon, step):
    dense_parts = [PolygonDense(polygon,step) for polygon in multi_polygon]
    return MultiPolygon(dense_parts)

def compute_voronoi_polygons(gdf):
    
    # Increase the density of points per edge to better define voronoi regions
    point_gdfs = []

    for i, r in progress_bar(gdf.iterrows(), total=gdf.shape[0]):

        points = list(zip(*PolygonDense(r.geometry, 1).exterior.coords.xy))
    
        point_gdf = gpd.GeoDataFrame(geometry=[Point(p) for p in points], crs=gdf.crs)

        point_gdf['ref_poly'] = i # Save reference polygon information

        point_gdfs.append(point_gdf)

    point_gdfs = pd.concat(point_gdfs)

    # Create voronoi regions
    smp = point_gdfs.unary_union
    smp_vd = svd(smp)
    gs = gpd.GeoSeries([smp_vd]).explode(index_parts=True)
    gdf_vd_primary = gpd.geodataframe.GeoDataFrame(geometry=gs, crs=gdf.crs)
    gdf_vd_primary.reset_index(drop=True)
    gdf['voronoi'] = gpd.sjoin(gdf_vd_primary, point_gdfs, how='inner', predicate='intersects').groupby('ref_poly').apply(lambda x : x.unary_union)
    
    return gdf

def compute_object_metrics(eval_y_true_shp, level_1, voronoi_shp, y_pred_shp, buffer=None, buffer_kind='outer'):
    """
    Compute metrics for a given reference polygon (eval_y_true_shp) and its corresponding predicted polygon (eval_y_pred_shp)
    within an evaluation region (eval_voronoi_shp). Metrics can be relaxed applying a tolerance buffer (buffer).
    """
          
    # Get corresponding voronoi polygon and prediction polygon
    eval_voronoi_shp = voronoi_shp[voronoi_shp.level_1 == level_1].unary_union
    eval_y_pred_shp = y_pred_shp[y_pred_shp.intersects(eval_voronoi_shp)].unary_union
    
    m = {
        'iou_pos': 0., 'dice_pos': 0., 'ppv_pos': 0., 'tpr_pos': 0., 'tnr_pos': 0., 'fpr_pos': 0., 'fnr_pos': 0., 'auc_pos': 0., 'kappa_pos': 0., 'mcc_pos': 0.,
        'eval_y_true_shp_pos': eval_y_true_shp, 'eval_y_pred_shp_pos': None, 'eval_buffer_shp_pos': None,
        'iou_neg': 0., 'dice_neg': 0., 'ppv_neg': 0., 'tpr_neg': 0., 'tnr_neg': 0., 'fpr_neg': 0., 'fnr_neg': 0., 'auc_neg': 0., 'kappa_neg': 0., 'mcc_neg': 0.,
        'eval_y_true_shp_neg': None, 'eval_y_pred_shp_neg': None, 'eval_buffer_shp_neg': None, 
        'eval_voronoi_shp': eval_voronoi_shp,
        'class_weights': None,
        'area': eval_y_true_shp.area
    }
        
    if eval_y_pred_shp and eval_y_pred_shp.area > 0: # Check if eval_y_pred_shp is not Nonetype due to empty intersection
        # Clip predicted polygon and reference ground truth with evaluated voronoi polygon
        inner_eval_y_true_shp = eval_y_true_shp.intersection(eval_voronoi_shp)
        inner_eval_y_pred_shp = eval_y_pred_shp.intersection(eval_voronoi_shp)
        inner_eval_not_y_true_shp = eval_voronoi_shp.difference(inner_eval_y_true_shp)
        inner_eval_not_y_pred_shp = eval_voronoi_shp.difference(inner_eval_y_pred_shp)
        if inner_eval_y_pred_shp and inner_eval_y_pred_shp.area > 0:
            # Buffer
            buff = get_buffer_shp(gpd.GeoDataFrame(geometry=[inner_eval_y_true_shp], crs=voronoi_shp.crs), buffer, kind=buffer_kind) if buffer else None
            
            # Compute TPs, TNs, FPs and FNs
            tp_pos, tn_pos, fp_pos, fn_pos = compute_confusion_matrix_shp(
                gpd.GeoDataFrame({'geometry': [inner_eval_y_pred_shp]}, crs=y_pred_shp.crs), 
                gpd.GeoDataFrame({'geometry': [inner_eval_y_true_shp]}, crs=y_pred_shp.crs),
                buffer=buff,
                boundary=eval_voronoi_shp
            )
            tp_neg, tn_neg, fp_neg, fn_neg = compute_confusion_matrix_shp(
                gpd.GeoDataFrame({'geometry': [inner_eval_not_y_pred_shp]}, crs=y_pred_shp.crs), 
                gpd.GeoDataFrame({'geometry': [inner_eval_not_y_true_shp]}, crs=y_pred_shp.crs),
                buffer=buff,
                boundary=eval_voronoi_shp
            )
            
            class_weights = np.array([inner_eval_y_true_shp.area, inner_eval_not_y_true_shp.area])
            class_weights /= class_weights.sum()

            m.update({
                'iou_pos': iou(tp_pos, tn_pos, fp_pos, fn_pos),
                'dice_pos': dice(tp_pos, tn_pos, fp_pos, fn_pos),
                'ppv_pos': ppv(tp_pos, tn_pos, fp_pos, fn_pos),
                'tpr_pos': tpr(tp_pos, tn_pos, fp_pos, fn_pos),
                'tnr_pos': tnr(tp_pos, tn_pos, fp_pos, fn_pos),
                'fpr_pos': fpr(tp_pos, tn_pos, fp_pos, fn_pos),
                'fnr_pos': fnr(tp_pos, tn_pos, fp_pos, fn_pos),
                'auc_pos': auc(tp_pos, tn_pos, fp_pos, fn_pos),
                'kappa_pos': kappa(tp_pos, tn_pos, fp_pos, fn_pos),
                'mcc_pos': mcc(tp_pos, tn_pos, fp_pos, fn_pos),
                'eval_y_true_shp_pos': inner_eval_y_true_shp,
                'eval_y_pred_shp_pos': inner_eval_y_pred_shp,
                'eval_buffer_shp_pos': buff,
                'iou_neg': iou(tp_neg, tn_neg, fp_neg, fn_neg),
                'dice_neg': dice(tp_neg, tn_neg, fp_neg, fn_neg),
                'ppv_neg': ppv(tp_neg, tn_neg, fp_neg, fn_neg),
                'tpr_neg': tpr(tp_neg, tn_neg, fp_neg, fn_neg),
                'tnr_neg': tnr(tp_neg, tn_neg, fp_neg, fn_neg),
                'fpr_neg': fpr(tp_neg, tn_neg, fp_neg, fn_neg),
                'fnr_neg': fnr(tp_neg, tn_neg, fp_neg, fn_neg),
                'auc_neg': auc(tp_neg, tn_neg, fp_neg, fn_neg),
                'kappa_neg': kappa(tp_neg, tn_neg, fp_neg, fn_neg),
                'mcc_neg': mcc(tp_neg, tn_neg, fp_neg, fn_neg),
                'eval_y_true_shp_neg': inner_eval_not_y_true_shp,
                'eval_y_pred_shp_neg': inner_eval_not_y_pred_shp,
                'eval_buffer_shp_neg': buff,
                'class_weights': class_weights,
            })
    
    return m