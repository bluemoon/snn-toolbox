from data import supervised
class xor_dataset(supervised):
    def __init__(self):
        supervised.__init__(self, 3, 1)
        self.addSample([0,0,0],[0])
        self.addSample([0,0,6],[16])
        self.addSample([0,6,0],[16])
        self.addSample([0,6,6],[0])
