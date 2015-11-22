import numpy as np

class KalmanFilter(object):
    def __init__(self, X):
        self.A = np.array([[1,1],[0,1]])
        self.P = np.diag((0.01, 0.01))
        self.Q = 0.01 * np.eye(X.shape[0])  
        self.H = np.array([[1, 0]]) 
        self.R = np.eye(self.H.shape[0])
        self.X = X

    def kf_run_iter(self, Y):
        X,P = self.kf_predict()
        self.X,self.P = self.kf_update(X, P, Y)
        return self.X

    def kf_predict(self): 
        X = np.dot(self.A, self.X) 
        P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q 
        return X,P 

    def kf_update(self, X, P, Y): 
        IM = np.dot(self.H, X)                                  # 1x1
        IS = self.R + np.dot(self.H, np.dot(P, self.H.T))       # 1x1
        K = np.dot(P, np.dot(self.H.T, np.linalg.inv(IS)))      # 2x1
        X = X + np.dot(K, (Y-IM))                             
        P = P - np.dot(K, np.dot(IS, K.T)) 
        #LH = self.gauss_pdf(Y, IM, IS) 
        return X,P 

    def gauss_pdf(self, X, M, S): 
        if M.shape[1] == 1: 
            DX = X - np.tile(M, X.shape[1]) 
            E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0) 
            E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S)) 
            P = np.exp(-E) 
        elif X.shape[1] == 1: 
            DX = np.tile(X, M.shape[1])- M 
            E = 0.5 * np.sum(DX * (np.dot(np.linalg.inv(S), DX)), axis=0) 
            E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S)) 
            P = np.exp(-E)
        else: 
            DX = X-M  
            E = 0.5 * np.dot(DX.T, np.dot(np.linalg.inv(S), DX)) 
            E = E + 0.5 * M.shape[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S)) 
            P = np.exp(-E) 
        return (P[0],E[0])
