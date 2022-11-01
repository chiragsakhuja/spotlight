import numpy as np
import sklearn.gaussian_process as gp
import sklearn.preprocessing as pp

import search_utils

class BO:
    def __init__(self, args, warmup_iters=10, exploration_ratio=0.1):
        if args.kernel == 'linear':
            self.kernel = gp.kernels.DotProduct() + gp.kernels.WhiteKernel()
        elif args.kernel == 'matern':
            self.kernel = gp.kernels.Matern() + gp.kernels.WhiteKernel()
        elif args.kernel == 'rbf':
            self.kernel = gp.kernels.RBF() + gp.kernels.WhiteKernel()
        self.model = gp.GaussianProcessRegressor(kernel=self.kernel)
        self.train_x = []
        self.train_y = []
        self.scale_x = pp.StandardScaler()
        self.scale_y = pp.StandardScaler()
        self.warmup = True
        self.warmup_iters = warmup_iters
        self.standarization = True
        self.exploration_ratio = exploration_ratio

    def update(self, xs, ys):
        self.train_x = np.copy(xs)
        self.train_y = [[y] for y in ys]
        self.warmup = len(self.train_x) <= self.warmup_iters

    def fit(self):
        if not self.warmup:
            if self.standarization:
                train_x_std = self.scale_x.fit_transform(self.train_x)
                train_y_std = self.scale_y.fit_transform(self.train_y)
                self.predictor = self.model.fit(train_x_std, train_y_std)
            else:
                self.predictor = self.model.fit(self.train_x, self.train_y)

    def predict(self, batch_x, return_std=True):
        if self.standarization:
            batch_x_std = self.scale_x.transform(batch_x)
            return self.predictor.predict(batch_x_std, return_std=return_std)
        else:
            return self.predictor.predict(batch_x, return_std=return_std)

    def run(self, batch_x):
        if self.warmup or np.random.random() < self.exploration_ratio:
            return list(range(len(batch_x)))
        preds, std = self.predict(batch_x=batch_x)
        sort_index = np.argsort(preds - 1.0 * std)
        return sort_index

def generate_hw_batch(hw_space, batch_size):
    hw_points = list()
    hw_feats = list()
    for _ in range(batch_size):
        space_idx = np.random.randint(hw_space.size)
        hw_point = hw_space.build_point(space_idx)
        hw_feat = search_utils.get_hw_point_feats(hw_point, hw_space.num_levels)
        hw_points.append(hw_point)
        hw_feats.append(hw_feat)
    return hw_points, hw_feats

def generate_sw_batch(sw_space, hw_point, batch_size, excluded_feats, dataflow):
    sw_points = list()
    sw_feats = list()
    for _ in range(batch_size):
        space_idx = np.random.randint(sw_space.size)
        sw_point = sw_space.build_point(space_idx)
        sw_feat = search_utils.get_sw_point_feats(hw_point, sw_point, sw_space.num_levels, excluded_feats, dataflow)
        sw_points.append(sw_point)
        sw_feats.append(sw_feat)
    return sw_points, sw_feats