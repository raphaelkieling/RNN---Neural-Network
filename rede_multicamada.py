#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:26:10 2018

@author: kieling
"""

import numpy as np

def sigmoid (soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)


entradas = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])
    
saidas = np.array([[0],[1],[1],[0]])

#pesos0 = np.array([[-0.424, -0.740,-0.961],[0.358, -0.577,-0.469]])
#pesos1 = np.array([[-0.017], [-0.893],[0.148]])
    
pesos0 = np.random.random((2,3))
pesos1 = np.random.random((3,1))
    

epocas = 1000

taxaAprendizagem = 1
momento = 1

for j in range(epocas):
    camadaEntrada = entradas
    #Ativaçao primeira camada
    somaSinapse0 = np.dot(camadaEntrada,pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    #Ativaçao segunda camada
    somaSinapse1 = np.dot(camadaOculta,pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    #Calculando o erro (cost function)
    erroCamadaSaida = saidas - camadaSaida
    ##encontra a media absoluta pois temos varias entradas tendo a media.
    ##diferente de quando e apenas um peso por entrada que apenas
    ## temos erro = esperado - saida_ativacao
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print('Erro: '+str(mediaAbsoluta))
    ##Derivada
    ##DeltaSaida
    ##
    ##Derivada para calcular o gradiente
    derivadaSaida = sigmoidDerivada(camadaSaida)
    ##DeltaSaida para calcular o gradiente
    deltaSaida = erroCamadaSaida * derivadaSaida
    #a transportadora colocar as colunas em uma linha transporta
    pesos1Transposta = pesos1.T 
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)