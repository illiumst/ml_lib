import numpy as np


class FarthestpointSampling():

    def __init__(self, K):
        self.k = K

    @staticmethod
    def calc_distances(p0, points):
        return ((p0[:3] - points[:, :3]) ** 2).sum(axis=1)

    def __call__(self, pts, *args, **kwargs):

        if pts.shape[0] < self.k:
            return pts

        farthest_pts = np.zeros((self.k, pts.shape[1]))
        farthest_pts_idx = np.zeros(self.k, dtype=np.int)
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, self.k):
            farthest_pts_idx[i] = np.argmax(distances)
            farthest_pts[i] = pts[farthest_pts_idx[i]]

            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))

        return farthest_pts_idx
