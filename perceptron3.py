#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:58:45 2018

@author: kieling
"""
import numpy as np

entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([0,1,1,1])
pesos = np.array([0.0,0.0])
taxaDeAprendizagem = 0.1

def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

def calculaSaida(data):
    s = data.dot(pesos)
    return stepFunction(s)

def treinar():
    erro = 1
    while(erro !=0):
        erro = 0
        for i in range(len(saidas)):
            somatorioAtivacao = calculaSaida(np.asarray(entradas[i]))
            erro  += abs(saidas[i] - somatorioAtivacao )
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaDeAprendizagem * entradas[i][j] * erro)
                #print('Peso atualizado' + str(pesos[j]))
        #print('Erro total' + str(erro))
        
treinar()

print('Pesos atualizados' + str(pesos))
print(calculaSaida(np.array([0,1])))