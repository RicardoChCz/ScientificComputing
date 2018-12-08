# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 05:14:51 2017

@author: Ricardo Chávez Cáliz
"""
import numpy as np
from matplotlib.mlab import bivariate_normal
from matplotlib import pylab as plt


# Matriz de covarianza
def matrizCovarianza(s1,s2,pho):
    """
    
    """
    return np.array([[s1*s1, pho*s1*s2],
                     [pho*s1*s2, s2*s2]])
    
def modify(v,e,alpha):
    """
    Modificar eigenvalores N_{i} = alpha*raíz de lambda_{1} * V_{i}
    """
    return np.array([v[0] * alpha * np.sqrt(abs(e)), 
                     v[1] * alpha * np.sqrt(abs(e)) ])
    
def limites(V1, V2):
    """
    Define los límites considerando la mayor proyección de los vectores modificados
    N1 y N2 en x y en y respectivamente.
    """
    return max(abs(V1[0]), abs(V2[0])), max(abs(V1[1]), abs(V2[1]))
    
def vectorGauss(m,sigma):
    k=len(m)
    
    #Aplicar Cholesky
    U = np.linalg.cholesky(sigma)
    
    #Generar vector aleatorio de dimensión k con distribución N(0,1)
    Z = np.zeros([k, 1], dtype=float)
    for i in range(0, k):
        Z[i]=np.random.normal(0,1)

    #Generar vector aleatorio de dimensión 1,k con ditribución N(m, sigma)
    
    X = m + np.dot(np.transpose(U),Z)
    return X


def muestraNMV(m,sigma,n):
    k=len(m)
    A=np.zeros((n,k))
    for i in range(0, n):
        X=vectorGauss(m,sigma)
        for j in range(0,k):
            A[i,j] = X[j,0]
    return A

def grafBivariada(m,sigma):
    """
    Función que grafica contornos de nivel  de bivariada
    """
    
    EigVal,EigVect = np.linalg.eig(sigma)

    # Construir eigenvectores normalizados
    V1,V2 = EigVect
    
    # Modificar eigenvalores N_{i} = alpha*raíz de lambda_{1} * V_{i}
    alpha = 3
    V1 = modify(V1,EigVal[0],alpha)
    V2 = modify(V2,EigVal[1],alpha)
    (lx,ly) = limites(V1,V2)

    #Dominio de graficación X Y usando los límites y trasladando a la media.
    x = np.arange(-lx + m[0], lx + m[0], 0.1)
    y = np.arange(-ly + m[1], ly + m[1], 0.1)
    X,Y = np.meshgrid(x, y)
    
    #Bivariada con dominio X Y, sigma1, sigma2, mu1, mu2, pho*s1*s2.
    Z = bivariate_normal(X, Y, s1, s2, m[0], m[1], sigma[0,1])
    
    plt.contour(X,Y,Z, alpha=0.5)
    
    
def graficaMuestraBi(m,sigma,n):
    """
    Grafica una muestra normal bivriada de tamaño n con parámetros m y sigma
    usando el metodo de MH con kerneles híbridos de parametro w1, junto con 
    los contornos de nivel de la densidad correspondiente.
    Input: array, array, float, int (media, matriz de covarianza, probabiidad de
           tomar el primer Kernel, tamaño)
    Output: Muestra gráfica
    """    
    M1 = muestraNMV(m,sigma,n)
    
    A= M1[:,0]
    B= M1[:,1]
    x = (A).tolist()
    y = (B).tolist()

    #Scatter
    colors = np.arange(0, 1, 1.0/n)
    area = 50*np.ones(n)
    plt.scatter(x, y, s=area, c=colors, alpha=0.8)
    grafBivariada(m,sigma)
    plt.show()
    

if __name__ == "__main__":
    #Incializar variables aleatorias.
    pho = np.random.uniform(-1, 1)
    s1 = np.random.uniform(1, 3)
    s2 = np.random.uniform(1, 3)

    k=2
    m = np.zeros([k, 1], dtype=float)
    for i in range(0, k):
        m[i]=np.random.uniform(-10, 10)
    
    sigma = matrizCovarianza(s1,s2,pho)
    
    graficaMuestraBi(m, sigma, 1000)
