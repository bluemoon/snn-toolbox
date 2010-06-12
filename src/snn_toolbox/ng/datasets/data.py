from numpy import *

class data:
    def __init__(self):
        self.data = {}
        self.endmarker = {}
        self.link = []
        self.index = 0
        self.vectorformat = 'none'
        self._convert = lambda(x):x
        
    def reset(self):
        self.index = 0

    def addField(self, label, dim):
        self.data[label] = zeros((0, dim), float)
        self.endmarker[label] = 0

    def linkFields(self, linklist):
        length = self[linklist[0]].shape[0]
        for l in linklist:
            if self[l].shape[0] != length:
                raise Exception
            
        self.link = linklist

    def getDimension(self, label):
        """Return the dimension/number of columns for the field given by 
        `label`."""
        try:
            dim = self.data[label].shape[1]
        except KeyError:
            raise KeyError('dataset field %s not found.' % label)
        return dim
    def _appendUnlinked(self, label, row):
        """Append `row` to the field array with the given `label`. 
        
        Do not call this function from outside, use ,append() instead. 
        Automatically casts vector to a 2d (or higher) shape."""
        if self.data[label].shape[0] <= self.endmarker[label]:
            self._resize(label)
         
        self.data[label][self.endmarker[label], :] = row
        self.endmarker[label] += 1
    def append(self, label, row):
        """Append `row` to the array given by `label`. 
        
        If the field is linked with others, the function throws an 
        `OutOfSyncError` because all linked fields always have to have the same
        length. If you want to add a row to all linked fields, use appendLink 
        instead."""
        if label in self.link:
            raise OutOfSyncError
        self._appendUnlinked(label, row)
            
    def appendLinked(self, *args):
        """Add rows to all linked fields at once."""
        assert len(args) == len(self.link)
        for i, l in enumerate(self.link):
            self._appendUnlinked(l, args[i])
            
    def getLinked(self, index=None):
        """Access the dataset randomly or sequential.
        
        If called with `index`, the appropriate line consisting of all linked
        fields is returned and the internal marker is set to the next line. 
        Otherwise the marked line is returned and the marker is moved to the
        next line."""
        if self.link == []:
            raise NoLinkedFieldsError('The dataset does not have any linked fields.')
            
        if index == None:
            # no index given, return the currently marked line and step marker one line forward
            index = self.index
            self.index += 1
        else:
            # return the indexed line and move marker to next line
            self.index = index + 1
        if index >= self.getLength():
            raise IndexError('index out of bounds of the dataset.')
            
        return [self._convert(self.data[l][index]) for l in self.link]

        
    def __getitem__(self, field):
        return self.getField(field)
        
    def __iter__(self):
        self.reset()
        while not self.endOfData():
            yield self.getLinked()
    def endOfData(self):
        return self.index == self.getLength()
    
    def getField(self, label):
        """Return the entire field given by `label` as an array or list,
        depending on user settings."""
        if self.vectorformat == 'list':
            return self.data[label][:self.endmarker[label]].tolist()
        else:
            return self.data[label][:self.endmarker[label]]
        
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
        """Increase the buffer size. It should always be one longer than the
        current sequence length and double on every growth step."""
        shape = list(a.shape)
        shape[0] = (shape[0] + 1) * 2
        return resize(a, shape)
    
    
class supervised(data):
    def __init__(self, inp, target):
        data.__init__(self)
        self.addField('input', inp)
        self.addField('target', target)
        self.linkFields(['input', 'target'])
        self.index = 0
        
        # the input and target dimensions
        self.indim = self.getDimension('input')
        self.outdim = self.getDimension('target')
        
    def addSample(self, inp, target):
        self.appendLinked(inp, target)
    def _provideSequences(self):
        return iter(map(lambda x: [x], iter(self)))
