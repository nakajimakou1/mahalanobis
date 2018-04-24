
import numpy as np
import numpy
import tqdm

import matplotlib.pyplot as plt


try:
    import cupy as cp
    xp = cp
    print('# use cupy')
except:
    xp = np
    print('# use numpy')


def getInverse(matrix):
    # unit mat
    fea_dim = len(matrix)
    inv_a = xp.array([[1.0 if i==j else 0.0 for i in range(fea_dim)] for j in range(fea_dim)], dtype=xp.float64)

    if not xp.linalg.det(matrix)==0:
        for i in range(fea_dim):
            if matrix[i][i]==0:
                for k in range(i, fea_dim):
                    if not matrix[k][i]==0:
                        tmp = matrix[k, :].copy()
                        matrix[k, :] = matrix[i, :].copy()
                        matrix[i, :] = tmp
                        tmp = inv_a[k, :].copy()
                        inv_a[k, :] = inv_a[i, :].copy()
                        inv_a[i, :] = tmp
                        break

            buf = 1.0 / matrix[i][i]
            inv_a[i, :] = inv_a[i, :] * buf
            matrix[i, :] = matrix[i, :] * buf
            for j in range(fea_dim):
                if not i==j:
                    buf =  matrix[j][i].copy()
                    inv_a[j,:] = inv_a[j,:] - inv_a[i,:] * buf
                    matrix[j,:] = matrix[j,:] - matrix[i,:] * buf

    else:
        print('det:', xp.linalg.det(matrix))
        print(' so use unit matrix instead')

    return inv_a

class mahalanobis:
    
    def __init__(self, _trains, _targets):
        print('# setup mahalanobis machine learning')
        self.trains = xp.array(_trains, dtype=xp.float64)
        self.targets = xp.array(_targets, dtype=xp.int32)
        self.calc_convariance_matrix()

    def calc_convariance_matrix(self):
        print('# calculate inverse of convariance matrix')
        self.classnum = int(max(self.targets))+1
        fea_dim = len(self.trains[0])

        trains_summed_by_class = [] # trainSummedByClass[0] -> train[targets==0]
        self.mean_vec = [] # self.mean_vec[0] -> xp.mean(train[targets==0], axis=0)
        for c in range(self.classnum):
            trains_summed_by_class.append(self.trains[self.targets==c])
            self.mean_vec.append(xp.mean(trains_summed_by_class[-1], axis=0))

        self.convariance_matrixs = [] # cm[0] -> convariance_matrix of targets==0
        self.inv_convariance_matrixs = []
        for c in range(self.classnum):
            convariance_matrix = []
            for i in range(fea_dim):
                convariance_vec = []
                for j in range(fea_dim):
                    xi_mi = trains_summed_by_class[c].T[i] - self.mean_vec[c][i]
                    xj_mj = trains_summed_by_class[c].T[j] - self.mean_vec[c][j]
                    convariance_vec.append(  xp.sum( xi_mi * xj_mj, axis=0) / len(self.targets[self.targets==c])  )

                convariance_matrix.append( convariance_vec )
            
            self.convariance_matrixs.append( convariance_matrix )

            convariance_matrix = xp.array(convariance_matrix, dtype=xp.float64)
            print('# class {}, convariance_matrix has been created'.format(c))

            inv_a = getInverse(convariance_matrix.copy())

            self.inv_convariance_matrixs.append(inv_a)


    def pred(self, tests):
        print('start prediction')
        tests = xp.array(tests, dtype=xp.float64)
        possibilities = []
        for test in tqdm.tqdm(tests):
            distToClasses = []
            for c in range(self.classnum):
                disttmp = xp.dot((test-self.mean_vec[c]).T, self.inv_convariance_matrixs[c])
                dist = xp.dot(disttmp, test-self.mean_vec[c])
                distToClasses.append(dist)
            
            possibilities.append(distToClasses)

        possibilities = xp.array(possibilities, dtype=xp.float64)
        return possibilities




def main():
   

    c1num = 100
    xc1 = np.random.randn(c1num) * 10 + 70
    yc1 = np.random.randn(c1num) * 15 + 40
    c2num = 100
    xc2 = np.random.randn(c2num) * 5 + 30
    yc2 = np.random.randn(c2num) * 10 + 30
    c3num = 100
    xc3 = np.random.randn(c3num) * 3 + 40
    yc3 = np.random.randn(c3num) * 7 + 70

    xtest = np.random.rand(1) * 30 + 30
    ytest = np.random.rand(1) * 30 + 30


    trains = np.r_[np.array([xc1, yc1]).T, np.array([xc2, yc2]).T, np.array([xc3,yc3]).T]
    targets = xp.r_[np.zeros(c1num), np.ones(c2num), np.ones(c3num)+1]
    print('trains.shape:', trains.shape)
    print('targets.shape:', targets.shape)

    plt.plot(xc1, yc1, 'r.', label='class 0')
    plt.plot(xc2, yc2, 'b.', label='class 1')
    plt.plot(xc3, yc3, 'g.', label='class 2')
    plt.plot(xtest, ytest, 'k*', label='class 4')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.savefig('test.png')

    
    tests = np.array([xtest, ytest]).T
    print('tests.shape:', tests.shape)

    myMah = mahalanobis(trains, targets)
    print('# mahalanobis distance')
    print(myMah.pred(tests))

    print('# euclidean distance')
    myMah.inv_convariance_matrixs[0] = xp.array([[1.0 if i==j else 0.0 for i in range(2)] for j in range(2)], dtype=xp.float64)
    myMah.inv_convariance_matrixs[1] = xp.array([[1.0 if i==j else 0.0 for i in range(2)] for j in range(2)], dtype=xp.float64)
    myMah.inv_convariance_matrixs[2] = xp.array([[1.0 if i==j else 0.0 for i in range(2)] for j in range(2)], dtype=xp.float64)
    print(xp.sqrt(myMah.pred(tests)))


if __name__=='__main__':
    main()
