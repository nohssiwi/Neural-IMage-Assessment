import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt

class RunningMetric(object):
    def __init__(self, metric_type):
        self._metric_type = metric_type
        if self._metric_type == 'MULTI':
            # RMSE SPCC PCC ACC
            self.gt = []
            self.pred = [] 

    def calculate_score(self, dis):
        weights = np.array([1, 2, 3, 4 ,5])
        return np.sum(dis * weights, axis=1)

    # def MSE(self, pred, gt):
    #     return np.square(pred - gt).mean()
    
    # def RMSE(self, pred, gt):
    #     rmse = sqrt(MSE(pred, gt))
    #     return rmse 
    
    def RMSE(self, pred, gt):
        rmse = sqrt(mean_squared_error(pred, gt))
        return rmse  

    def accuracy(self, pred, gt):
        pred_ge_3 = pred >= 3
        gt_ge_3 = gt >= 3
        return np.sum(pred_ge_3 ==  gt_ge_3) / pred.shape[0]

    def reset(self):
        if self._metric_type == 'MULTI':
            # RMSE SPCC PCC ACC
            self.gt = []
            self.pred = []

    def update(self, pred, gt):
        if self._metric_type == 'MULTI':
            gt_score = self.calculate_score(gt.data.cpu().numpy().reshape(-1,5)).tolist()
            self.gt += gt_score
            pred_score = self.calculate_score(pred.data.cpu().numpy().reshape(-1,5)).tolist()
            self.pred += pred_score

        
    def get_result(self):
        if self._metric_type == 'MULTI':
            pred = np.array(self.pred)
            gt = np.array(self.gt)
            return {
                'rmse': self.RMSE(pred, gt),
                'spcc': stats.spearmanr(pred, gt)[0],
                'plcc': stats.pearsonr(pred, gt)[0],
                'acc': self.accuracy(pred, gt)
            }

def get_metrics(params):
    met = RunningMetric(metric_type='MULTI')
    return met