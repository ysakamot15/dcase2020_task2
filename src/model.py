import numpy as np
from sklearn.decomposition import PCA
import scipy as sci

class mahalanobis_dist_model():
    def __init__(self):
        self.feature_name = 'maha' 
        pass

    def fit(self, train_data):
        ms = train_data[self.feature_name]
        self.invC = np.linalg.inv(np.cov(ms.T))
        self.mu = np.mean(ms, axis = 0)
        self.train_data_scores = np.zeros(ms.shape[0])
        for i in range(ms.shape[0]):
            self.train_data_scores[i] = self.score({self.feature_name:ms[i, :]})

    def score(self, test_data):
        test_m = test_data[self.feature_name]
        return np.sqrt((test_m - self.mu) @ self.invC @ (test_m - self.mu))

class subspace_dist_model():
    def __init__(self, k):
        self.feature_name = 'ssd'
        self.k = k
 
    def fit(self, train_data):
        Ws = train_data[self.feature_name]
        self.subspace_basis = Ws
        self.train_data_scores = np.zeros(Ws.shape[0])
        for i in range(Ws.shape[0]):
            self.train_data_scores[i] = self.score({self.feature_name:Ws[i, :, :]}, remove_first = True)

    # when used for train data scoring, 
    # be "removed first = True" because nearest one is itself.
    def score(self, test_data, remove_first = False):
        test_W = test_data[self.feature_name]
        norms = np.linalg.norm(self.subspace_basis.transpose(0, 2, 1) @ test_W, 2, axis = (1, 2))
        all_dists = (1 - norms) * ( 1 + norms)
        
        remove = 0
        if remove_first == True:
            remove = 1
        knn_mean_dist = np.mean(np.sort(all_dists)[remove:self.k + remove])
        return knn_mean_dist

class matrix_normal_distribution_model():
    def __init__(self, is_U_eye, feature_name, iter_num = 15, batch_size = 500):
        self.name = feature_name
        self.is_U_eye = is_U_eye
        self.iter_num = iter_num
        self.feature_name = feature_name
        self.batch_size = batch_size
        if is_U_eye == True:
            self.iter_num = 1
    
    def fit(self, train_data):
        Xs = train_data[self.feature_name]
        self.M = np.mean(Xs, axis = 0)
        N = Xs.shape[0]
        H = Xs.shape[1]
        W = Xs.shape[2]
        U = np.eye(H) 
        V = np.eye(W) 
        self.invU = np.eye(H) 
        self.invV = np.eye(W) 
        n_batches = N//self.batch_size
        if (N % self.batch_size != 0):
            n_batches += 1
        
        for i in range(self.iter_num):
            if self.is_U_eye == True:
                self.invU = np.eye(H) 
            else:
                Utmp = np.zeros((H, H))    
                for j in range(n_batches):
                    X_batch = Xs[j * self.batch_size:min(N, (j + 1) * self.batch_size), :, :] 
                    Utmp += np.sum((X_batch - self.M) @ self.invV @ (X_batch - self.M).transpose(0,2,1), axis = 0)
                
                U = Utmp/(N * W)
                if(np.linalg.matrix_rank(U) < U.shape[0]):
                    self.invU = np.linalg.pinv(U)
                else:
                    self.invU = np.linalg.inv(U)
            
            Vtmp = np.zeros((W, W))
            for j in range(n_batches):
                X_batch = Xs[j * self.batch_size:min(N, (j + 1) * self.batch_size), :, :] 
                Vtmp += np.sum((X_batch - self.M).transpose(0,2,1) @ self.invU @ (X_batch - self.M), axis = 0)
            V = Vtmp/(N * H)
            if(np.linalg.matrix_rank(V) < V.shape[0]):
                self.invV = np.linalg.pinv(V)
            else:
                self.invV = np.linalg.inv(V)

        self.train_data_scores = np.zeros(Xs.shape[0])
        for i in range(Xs.shape[0]):
            self.train_data_scores[i] = self.score({self.feature_name:Xs[i, :, :]})


    def score(self, test_data):
        test_X = test_data[self.feature_name]
        # if input length are shoter than train data length,
        # interporated with train mean.
        # if input length are longer than train data length, 
        # protruding part are deleted.
        if(test_X.shape[1] < self.M.shape[1]):
            X_ = self.M
            X_[:, :test_X.shape[1]] = test_X
        elif(test_X.shape[1] > self.M.shape[1]):
            X_ = test_X[:, :self.M.shape[1]]
        else:
            X_ = test_X
        Z = (X_ - self.M)
        return np.sqrt(np.trace(self.invV @ (Z).T @ self.invU @ (Z)))


def pca_decompose(X, d):
    pca = PCA(n_components= d)            
    Z = pca.fit_transform(X.T).T
    W = pca.components_.T
    m = pca.mean_
    Xd = W @ Z
    return Z, W, m, Xd

class total_model():
    def __init__(self, d, use_model_list, take_decimate, use_for_decimate_list):
        self.d = d
        self.take_decimate = take_decimate
        self.use_model_list = use_model_list
        self.use_for_decimate_list = use_for_decimate_list

    def fit(self, X):
        Ws = np.zeros((X.shape[0], X.shape[1], self.d)) 
        Zs = np.zeros((X.shape[0], self.d,  X.shape[2]))
        Xds = np.zeros((X.shape[0], X.shape[1],  X.shape[2]))
        ms  = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            Zs[i, :, :], Ws[i, :, :], ms[i, :], Xds[i, :, :] = pca_decompose(X[i, :, :], self.d)
        train_data = {'maha':ms, 'ssd':Ws, 'mndZ':Zs, 'mndX':Xds}

        is_leave = np.ones(X.shape[0], dtype=bool)
        all_train_data_scores = np.zeros((X.shape[0], len(self.use_model_list)))

        for i, use_model in enumerate(self.use_model_list):
            use_model.fit(train_data)
            all_train_data_scores[:, i] = use_model.train_data_scores
            median_score = np.median(use_model.train_data_scores)
            if self.take_decimate == True and self.use_for_decimate_list[i] == True:
                is_leave &= (use_model.train_data_scores <= median_score)

        self.leave_train_score = all_train_data_scores[is_leave, :]
        self.removed_train_score = all_train_data_scores[np.logical_not(is_leave), :]
        self.score_mean = np.mean(self.leave_train_score, axis = 0)
        self.score_invcov = np.linalg.inv(np.cov(self.leave_train_score.T))

    def score(self, X):
        Z, W, m, Xd = pca_decompose(X, self.d)
        test_data = {'maha':m, 'ssd':W, 'mndZ':Z, 'mndX':Xd}
        test_data_scores = np.zeros(len(self.use_model_list))
        for i, use_model in enumerate(self.use_model_list):
            test_data_scores[i] = use_model.score(test_data)


        Z = test_data_scores - self.score_mean
        Z[Z < 0] = 0
        return Z @ self.score_invcov @ Z, test_data_scores
