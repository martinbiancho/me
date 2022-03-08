""" GUILHERME KOHUT MARINBIANCHO """

import numpy as np
import matplotlib.pyplot as plt

# activation function
def f(z):
    return 0.5 + 0.5*np.tanh(z/2)

def df(z):
    return 0.25 - 0.25*np.tanh(z/2)**2

# rede neural
def rna(x, W, f=f):
    x = np.concatenate((np.array([1]), # (m_X+1 x 1)
                        x),
                        axis=0)
    return f(W.T @ x) # ---> ESCALAR

# training (Exercício)
def train(X, Y, f=f, df=df, params={}):
    max_iters = {}.get('max_iters', 300) # max iterations
    eta = params.get('eta', 0.1) # training step size

    # dimensions
    m_X, n_X = X.shape # n_X = l
    m_Y, n_Y = Y.shape # m_Y = l

    W = np.random.uniform(size=(m_X+1, m_Y)) # init W
    print(W.round(3))

    # insert bias dimension
    X = np.concatenate((np.ones((1, n_X)),
                        X),
                        axis=0)

    """ EXERCÍCIO """
    # PLOT
    iters = np.array([])
    values = np.array([])

    W0 = np.copy(W) # save W copy
    

    count = 1
    while count < max_iters:

        J = np.zeros((m_X + 1, m_Y)) # J com dimensões (m_X+1 x m_Y)
        for sample in np.random.permutation(n_X): # (ou m_Y, ambos são o número de amostras "l") percorre todas as amostras de entrada
            Z = W0.T @ X[:, sample] # (m_Y x m_X+1) @ (m_X+1 x 1) em coluna ---> (m_Y x 1)

            for i in np.arange(m_X+1): # percorre todas as dimensoes de cada amostra de entrada
                for j in np.arange(m_Y): # percorre todas as dimensoes de cada amostra de saída
                    Zj = Z[j] # (ESCALAR)
                    J[i,j] += 2. * (Y[j, sample] - f(Zj)) * df(Zj) * X[i, sample]

            W = W0 + eta * J # atualiza pesos
            W0 = np.copy(W)

        count += 1

        # PLOT
        iters = np.append(iters, count)

        erro = np.array([ np.linalg.norm(Y[:,0] - f(W0.T @ X[:,0])), # E para amostra i = 0
                          np.linalg.norm(Y[:,1] - f(W0.T @ X[:,1])),
                          np.linalg.norm(Y[:,2] - f(W0.T @ X[:,2])),
                          np.linalg.norm(Y[:,3] - f(W0.T @ X[:,3])) ])

        values = np.concatenate([values, erro])
        # values.append(W[0, 0])
    """ EXERCÍCIO """

    return W, iters, values.reshape(-1, 4)


# test case: tabelas verdade de E e OU
X_train = np.array([ [ 1,  1, -1, -1],
                     [ 1, -1,  1, -1] ])

Y_train = np.array([ [+1,  0,  0,  0],
                     [+1, +1, +1,  0] ])

W, iters, values = train(X_train, Y_train, params = {'max_ters': 10000})

# print(W)
print()
print(rna(X_train[:,0], W).round(3))
print(rna(X_train[:,1], W).round(3))
print(rna(X_train[:,2], W).round(3))
print(rna(X_train[:,3], W).round(3))

fig, ax = plt.subplots()
ax.plot(iters, values)
ax.set_title('Error per iteration')
plt.show()
