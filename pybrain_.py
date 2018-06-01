from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

net = buildNetwork(2, 3, 1)
data = SupervisedDataSet(2, 1)

data.addSample((0, 0), (0))
data.addSample((0, 1), (1))
data.addSample((1, 0), (1))
data.addSample((1, 1), (0))

trainer = BackpropTrainer(net, data)

epoch = 10000

for i in range(epoch):
    trainer.train()

print(net.activate([0, 0]))
print(net.activate([1, 0]))
print(net.activate([0, 1]))
print(net.activate([1, 1]))
