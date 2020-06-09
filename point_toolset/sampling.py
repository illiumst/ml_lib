from abc import ABC

import numpy as np


class _Sampler(ABC):

    def __init__(self, K, **kwargs):
        self.k = K
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class RandomSampling(_Sampler):

    def __init__(self, *args, **kwargs):
        super(RandomSampling, self).__init__(*args, **kwargs)

    def __call__(self, pts, *args, **kwargs):
        rnd_indexs = np.random.choice(np.arange(pts.shape[0]), min(self.k, pts.shape[0]), replace=False)
        return rnd_indexs


class FarthestpointSampling(_Sampler):

    def __init__(self, *args, **kwargs):
        super(FarthestpointSampling, self).__init__(*args, **kwargs)

    @staticmethod
    def calc_distances(p0, points):
        return ((p0[:3] - points[:, :3]) ** 2).sum(axis=1)

    def __call__(self, pts, *args, **kwargs):

        if pts.shape[0] < self.k:
            return pts

        else:
            farthest_pts = np.zeros((self.k, pts.shape[1]))
            farthest_pts_idx = np.zeros(self.k, dtype=np.int)
            farthest_pts[0] = pts[np.random.randint(len(pts))]
            distances = self.calc_distances(farthest_pts[0], pts)
            for i in range(1, self.k):
                farthest_pts_idx[i] = np.argmax(distances)
                farthest_pts[i] = pts[farthest_pts_idx[i]]

                distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))

            return farthest_pts_idx
