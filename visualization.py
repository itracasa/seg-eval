import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
from itertools import product
import seaborn as sns
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

def get_iou_mask(y_pred, y_true, return_cmap=True):
    iou = np.zeros_like(y_true, dtype='uint8')
    iou[np.logical_and(y_true == 1, y_pred == 1)] = 1
    iou[np.logical_and(y_true == 0, y_pred == 1)] = 2
    iou[np.logical_and(y_true == 1, y_pred == 0)] = 3
    
    if return_cmap:
        cmap = ListedColormap(['xkcd:white', 'xkcd:lime green', 'xkcd:light red', 'xkcd:bright blue'])
        return iou, cmap
    
    return iou

def get_boundary(gdf):
    return box(*gdf.total_bounds)

def outer_geometry(gdf):
    boundary = get_boundary(gdf)
    outer_geom = gpd.GeoDataFrame({'geometry': [boundary.difference(gdf.iloc[0].geometry).buffer(0)]}, crs=gdf.crs) # Buffer 0 to avoid invalid geometries
    return outer_geom

def gen_grid(outer_bounds, inner_bounds, crs, gsd=1):
    minX, minY, maxX, maxY = outer_bounds
    _minX, _minY, _maxX, _maxY = inner_bounds
    
    # Create a fishnet
    x = np.arange(minX, maxX, gsd)
    y = np.arange(minY, maxY, gsd)

    # Subquery based on object's boundary
    x = x[(x >= _minX) & (x <= _maxX)]
    y = y[(y >= _minY) & (y <= _maxY)]
    
    # geoms = pd.DataFrame.from_records(product(x, y), columns=['x', 'y']).parallel_apply(
    #     lambda r : Point(r.x,r.y), axis=1).values.tolist()
    
    geoms = list(map(Point, product(x, y)))

    return gpd.GeoDataFrame(geoms, columns=['geometry']).set_crs(crs)

def gen_bivariate_plot(df, eval_attr):
    
    fig, axs = plt.subplots(2, 2, figsize=(5, 5), gridspec_kw={'hspace': 0.005, 'wspace': 0.005, 'width_ratios': [5, 1], 'height_ratios': [1, 5]})

    xlim = (df.area.min(), df.area.max())
    ylim = (-0.1, 1.1)

    # Upper part charts
    sns.kdeplot(x=df.area, ax=axs[0, 0], color="xkcd:black", log_scale=True, fill=True)
    axs[0, 0].set_xscale('log'); axs[0, 0].set_xlim(xlim) # axs[0,0].set_ylim(0, 1); 

    # Right part charts
    sns.kdeplot(y=(df[eval_attr] - df[eval_attr].min()) / (df[eval_attr].max() - df[eval_attr].min()), ax=axs[1, 1], color="#3271a5", fill=True)
    axs[1,1].set_ylim(ylim);

    blues_cmap = LinearSegmentedColormap.from_list('custom_blues', ['#3271a5', '#4181B1', '#5090BD', '#60A0C9', '#6FB0D4', '#7EBFE0', '#8DCFEC'])

    # KDE middle part
    values = np.vstack([df.area, df[eval_attr]])
    kernel = stats.gaussian_kde(values)(values)
    sns.scatterplot(x=m_df.area, y=m_df[eval_attr], hue=kernel, ax=axs[1,0], palette=blues_cmap, s=2, edgecolor='none', legend=False)
    axs[1, 0].set_ylim(ylim); axs[1, 0].set_ylabel(f'{eval_attr}', fontsize=14)
    axs[1, 0].set_xscale('log'); axs[1, 0].set_xlabel('Area [m2]', fontsize=14); axs[1, 0].set_xlim(xlim)

    axs[0,0].axis("off")
    axs[0,1].axis("off")
    axs[1,1].axis("off")
    plt.show()