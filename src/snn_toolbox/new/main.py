from layer import *
from network import *
from neuron import *
from dataset import *
from train import *

def main():

    i = neurons(3)
    h = neurons(5)
    o = neurons(1)
    
    n = network()
    n += e_hidden(i, h)
    n += e_out(h, o)
    
    x = xor_dataset()
    t = train(n)
    while True:
        print t.train(x)
    
    
