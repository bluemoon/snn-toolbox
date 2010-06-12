import numpy as np

class dataset:
    def __init__(self):
        self.data = {}
        self.endmarker = {}
        self.link = []
        self.index = 0
        self.vectorformat = 'none'
        self._convert = lambda(x):x
        
    def reset(self):
        self.index = 0

    def add_field(self, label, dim):
        self.data[label] = np.zeros((0, dim), float)
        self.endmarker[label] = 0
        
    def link_fields(self, linklist):
        length = self[linklist[0]].shape[0]
        for l in linklist:
            if self[l].shape[0] != length:
                raise Exception
            
        self.link = linklist
        
    def _append_unlinked(self, label, row):
        if self.data[label].shape[0] <= self.endmarker[label]:
            self._resize(label)
         
        self.data[label][self.endmarker[label], :] = row
        self.endmarker[label] += 1
        
    def append_linked(self, *args):
        for i, l in enumerate(self.link):
            self._append_unlinked(l, args[i])
            
    def __getitem__(self, field):
        return self.get_field(field)
        
    def __iter__(self):
        self.reset()
        while not self.endOfData():
            yield self.get_linked()
            
    def endOfData(self):
        return self.index == self.getLength()
    
    def get_field(self, label):
        return self.data[label][:self.endmarker[label]]
        
    def get_linked(self, index=None):
        if self.link == []:
            raise Exception
            
        if index == None:
            # no index given, return the currently marked line and step marker one line forward
            index = self.index
            self.index += 1
        else:
            # return the indexed line and move marker to next line
            self.index = index + 1
        if index >= self.getLength():
            raise IndexError('index out of bounds of the dataset.')
            
        return [self.data[l][index] for l in self.link]
    
    def getLength(self):
        if self.link == []:
            try:
                length = self.endmarker[max(self.endmarker)]
            except ValueError:
                return 0
            return length
        else:
            # all linked fields have equal length. return the length of the first.
            l = self.link[0]
            return self.endmarker[l]

    def _resize(self, label=None):
        if label:
            label = [label]
        elif self.link:
            label = self.link
        else:
            label = self.data
        
        for l in label:
            self.data[l] = self._resizeArray(self.data[l])
    
    def _resizeArray(self, a):
        shape = list(a.shape)
        shape[0] = (shape[0] + 1) * 2
        return np.resize(a, shape)
    
class supervised(dataset):
    def __init__(self, inp, target):
        dataset.__init__(self)
        self.add_field('input', inp)
        self.add_field('target', target)
        self.link_fields(['input', 'target'])
        self.index = 0
        
        
    def addSample(self, inp, target):
        self.append_linked(inp, target)
        
    def _provideSequences(self):
        return iter(map(lambda x: [x], iter(self)))
    
class xor_dataset(supervised):
    def __init__(self):
        supervised.__init__(self, 3, 1)
        self.addSample([0,0,0],[0])
        self.addSample([0,0,6],[16])
        self.addSample([0,6,0],[16])
        self.addSample([0,6,6],[0])
