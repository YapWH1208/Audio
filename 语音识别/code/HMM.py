import numpy as np

class HiddenMarkovModel:
    def forward(self, Q, V, A, B, X, PI):
        N = len(Q)
        M = len(X)
        alphas = np.zeros((N, M))
        T = M

        for t in range(T):
            idxXi = V.index(X[t])
            for i in range(N):
                if t == 0:
                    alphas[i][t] = PI[t][i] * B[i][idxXi]
                    print("alphas%d(%d) = pi%d * b%d(x1) = %f * %f = %f" % (t, i, i, i, PI[t][i], B[i][idxXi], alphas[i][t]))
                else:
                    alphas[i][t] = np.dot(
                        [alpha[t-1] for alpha in alphas],
                        [a[i] for a in A]
                    ) * B[i][idxXi]
                    print("alphas%d(%d) = [sigma alphas%d(i) * a%d%d] * b%d(x%d) = %f" %(t, i, t-1, i, i, i, t, alphas[i][t]))

        P = sum(alphas[:,-1])

        return P, alphas

    def backward(self, Q, V, A, B, X, PI):
        N = len(Q)
        M = len(X)
        betas = np.ones((N, M))

        for i in range(N):
            print("betas%d(%d) = 1" % (M, i))
        for t in range(M-2, -1, -1):
            idxXi = V.index(X[t+1])

            for i in range(N):
                betas[i][t] = np.dot(
                    np.multiply(A[i], [b[idxXi] for b in B]),
                    [beta[t+1] for beta in betas]
                )
                realT = t+1
                realI = i+1
                print("betas%d(%d) = sigma[a%dj * Bj(x%d) * beta%d(j)] = (" %(realT, realI, realI, realT+1, realT+1), end='')

                for j in range(N):
                    print("%.2f * %.2f * %.2f + " %(A[i][j], B[j][idxXi], betas[j][t+1]), end='')
                print("0) = %.3f" % betas[i][t])
        idxXi = V.index(X[0])
        P = np.dot(
            np.multiply(PI[0], [b[idxXi] for b in B]),
            [beta[0] for beta in betas]
        )
        print("P(X|lambda) = ", end='')
        for i in range(N):
            print("%.1f * %.1f * %.5f + " %(PI[0][i], B[i][idxXi], betas[i][0]), end='')
        print("0 = %f" % P)

        return P, betas
    
    def viterbi(self, Q, V, A, B, X, PI):
        N = len(Q)
        M = len(X)
        deltas = np.zeros((N, M))
        psis = np.zeros((N, M))
        Y = np.zeros((1, M))

        for t in range(M):
            realT = t+1
            idxXi = V.index(X[t])
            for i in range(N):
                realI = i+1
                if t == 0:
                    deltas[i][t] = PI[0][i] * B[i][idxXi]
                    psis[i][t] = 0
                    print("delta1(%d) = pi%d * b%d(x1) = %.2f * %.2f = %.2f" %(realI, realI, realI, PI[0][i], B[i][idxXi], deltas[i][t]))
                    print("psis1(%d) = 0" % realI)
                else:
                    deltas[i][t] = np.max(np.multiply([delta[t-1] for delta in deltas], [a[i] for a in A])) * B[i][idxXi]
                    print("delta%d(%d) = max[delta%d(j) * a%d%d] * b%d(x%d) = %.2f * %.2f = %.5f" %(realT, realI, realT-1, realI, realI, realI, realT, np.max(np.multiply([delta[t-1] for delta in deltas], [a[i] for a in A])), B[i][idxXi], deltas[i][t]))

                    psis[i][t] = np.argmax(np.multiply([delta[t-1] for delta in deltas], [a[i] for a in A])) + 1
                    print("psis%d(%d) = argmax[delta%d(j) * a%d%d] = %d" %(realT, realI, realT-1, realI, realI, psis[i][t]))
        
        Y[0][M-1] = np.argmax([delta[M-1] for delta in deltas]) + 1
        print("Y%d = argmax[deltaT(i)] = %d" %(M, Y[0][M-1]))
        for t in range(M-2, -1, -1):
            Y[0][t] = psis[int(Y[0][t+1])-1][t+1]
            print("Y%d = psis%d(Y%d) = %d" %(t+1, t+2, t+2, Y[0][t]))
        print("最优状态序列：", Y)
        
        return deltas, psis


"""
Q: 状态序列
V: 观测序列
A: 状态转移概率矩阵
B: 观测概率矩阵
X: 观测序列
PI: 初始状态概率向量
"""
Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
# X = ['红', '白', '红', '红', '白', '红', '白', '白']
X = ['红', '白', '红']
PI = [[0.2, 0.4, 0.4]]

hmm = HiddenMarkovModel()
P, alpha = hmm.forward(Q, V, A, B, X, PI)
print(P)
print(alpha)
print()
P, betas = hmm.backward(Q, V, A, B, X, PI)
print(P)
print(betas)
print()
deltas, psis = hmm.viterbi(Q, V, A, B, X, PI)
print(deltas)
print(psis)