import numpy as np


class FarthestpointSampling():

    def __init__(self, K):
        self.k = K

    def __call__(self, pts, *args, **kwargs):

        if pts.shape[0] < self.k:
            return pts

        def calc_distances(p0, points):
            return ((p0[:3] - points[:, :3]) ** 2).sum(axis=1)

        farthest_pts = np.zeros((self.k, pts.shape[1]))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = calc_distances(farthest_pts[0], pts)
        for i in range(1, self.k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))

        return farthest_pts
