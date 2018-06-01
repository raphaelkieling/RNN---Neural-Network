#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:48:44 2018

@author: kieling
"""

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

rede = buildNetwork(3,3,1)
base = SupervisedDataSet(3,1)

#pernacurta late corre
# 1 - porco 2 - cachorro 
base.addSample((1,0,0),(1))
base.addSample((1,0,1),(1))
base.addSample((0,1,1),(2))
base.addSample((1,1,1),(2))

treinamento = BackpropTrainer(rede, dataset=base, learningrate=0.01, momentum=0.06)

for i in range(1000):
    erro  = treinamento.train()
    print(erro)
        
print(np.around(rede.activate([1,0,0])))
print(np.around(rede.activate([1,0,1])))
print(np.around(rede.activate([0,1,1])))
print(np.around(rede.activate([1,1,1])))
print(np.around(rede.activate([0,1,0])))