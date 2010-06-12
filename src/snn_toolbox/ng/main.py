from threaded import *
from numpy import *
from connections.connections import *
from modules.linear_layer import linear_layer
from modules.e_hidden import e_hidden
from modules.e_output import e_output
from network.network import network
from datasets.xor import xor_dataset

def main():
    network = network_container()
    network += input_layer(3, 5)
    network += output_layer(5, 1)

    m = Manager()
    m += network
    m += Debugger()
    
    input, desired = xor(1)
    m.push(forward_pass(input, desired))
    
    m.run()
    
def build_network(*layers):

    
    n = network()
    n.addInputModule(linear_layer(layers[0], name='in'))
    for i, num in enumerate(layers[1:-1]):
        layername = 'hidden%i' % i
        n.addModule(linear_layer(num, name=layername))
        
    n.addOutputModule(linear_layer(layers[-1], name='out'))
    
    for i in range(len(layers)-3):
        n.addConnection(connection(n['hidden%i' % i], n['hidden%i' % (i + 1)]))
        
    if len(layers) == 2:
        n.addConnection(connection(n['in'], n['out']))
    else:
        n.addConnection(e_in_connection(n['in'], n['hidden0'], name='c1'))
        n.addConnection(e_out_connection(n['hidden%i' % (len(layers) - 3)], n['out'], name='c2'))
        #n.addConnection(connection(n.in, n.out))]

    n.sortModules()
    return n
                                              

def modular():
    from trainers.backprop import backprop
    
    n = build_network(3,5,1)
    b = backprop(n)
    b.train_until_convergence(xor_dataset())
    
    
    
    
    
    
