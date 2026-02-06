import numpy as np
from pyevtk.hl import gridToVTK
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree


def save_to_rectilinear_grid(pos, res, filename, attr_dict=None):
    if attr_dict is None:
        attr_dict = {}
    # TODO: export field data such as mean, variance
    x, y, z = (np.ascontiguousarray(pos.reshape(res, res, res, 3)[:, 0, 0, 0]),
                np.ascontiguousarray(pos.reshape(res, res, res, 3)[0, :, 0, 1]),
                np.ascontiguousarray(pos.reshape(res, res, res, 3)[0, 0, :, 2]))

    gridToVTK(filename, x, y, z,
              pointData={item: np.ascontiguousarray(attr_dict[item].reshape(res, res, res)) for item in attr_dict}
             )
    

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default `l2`
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_to_x\', \'x_to_y\', \'bi\'")
        
    return chamfer_dist


def chamfer_distance_v2(X, Y, p=2, reduction='sum'):
    """
    Chamfer distance between two point sets X (N x D) and Y (M x D). Optimized version using cKDTree.

    Parameters
    ----------
    p : int or float
        Minkowski norm parameter: 2 = Euclidean. (Use 1 for Manhattan if you really need it.)
    squared : bool
        If True and p==2, square the distances before averaging (CD-L2).
        If True with p!=2, this is nonstandard; usually leave squared=False for p!=2.
    reduction : {'sum', 'mean_both'}
        'sum'        -> mean(d(Y->X)) + mean(d(X->Y))     [common in papers]
        'mean_both'  -> 0.5 * (mean(d(Y->X)) + mean(d(X->Y)))

    Returns
    -------
    cd : float
        L1 and L2 Chamfer distance.
    """
    treeX = cKDTree(X)
    d_yx, _ = treeX.query(Y, k=1, p=p, workers=-1)  # distances Y->X
    treeY = cKDTree(Y)
    d_xy, _ = treeY.query(X, k=1, p=p, workers=-1)  # distances X->Y

    if reduction == 'sum':
        return d_yx.mean() + d_xy.mean(), (d_yx**2).mean() + (d_xy**2).mean()
    elif reduction == 'mean_both':
        return 0.5 * (d_yx.mean() + d_xy.mean()), 0.5 * ((d_yx**2).mean() + (d_xy**2).mean())
    else:
        raise ValueError("reduction must be 'sum' or 'mean_both'")

def normal_consistency(x, y, normal_x, normal_y, metric='l2', direction='bi'):
    """Normal consistency between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    normal_x: numpy array [n_points_x, 3]
                vertex normals of first point cloud
    normal_y: numpy array [n_points_y, 3]
                vertex normals of second point cloud
    metric: string or callable, default `l2`
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    normal_consistency: float
        computed bidirectional normal consistency:
    """
    
    if direction=='y_to_x':
        raise NotImplementedError
    elif direction=='x_to_y':
        raise NotImplementedError
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x, idx_y_to_x = x_nn.kneighbors(y)
        normal_y_to_x = normal_x[idx_y_to_x.squeeze()]
        nc_y_to_x = np.abs(np.sum(normal_y * normal_y_to_x, axis=-1)/(np.linalg.norm(normal_y,axis=-1)*np.linalg.norm(normal_y_to_x,axis=-1) + 1e-8))

        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y, idx_x_to_y = y_nn.kneighbors(x)
        normal_x_to_y = normal_y[idx_x_to_y.squeeze()]
        nc_x_to_y = np.abs(np.sum(normal_x * normal_x_to_y, axis=-1)/(np.linalg.norm(normal_x,axis=-1)*np.linalg.norm(normal_x_to_y,axis=-1) + 1e-8))

        normal_consistency = 0.5 * (np.mean(nc_y_to_x) + np.mean(nc_x_to_y))
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_to_x\', \'x_to_y\', \'bi\'")
        
    return normal_consistency

if __name__ == "__main__":
    # test chamfer distance
    x = np.random.rand(1000, 3)
    y = np.random.rand(2000, 3)
    cd = chamfer_distance(x, y)
    print("Chamfer distance L1:", cd)

    cd_v2, cd_v2_sq = chamfer_distance_v2(x, y)
    print("Chamfer distance v2 L1:", cd_v2)
    print("Chamfer distance v2 L2:", cd_v2_sq)

    # test normal consistency
    normal_x = np.random.rand(1000, 3)
    normal_y = np.random.rand(2000, 3)
    nc = normal_consistency(x, y, normal_x, normal_y)
    print("Normal consistency:", nc)