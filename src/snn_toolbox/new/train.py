class train:
    def __init__(self, module):
        self.module = module
        
    def train(self, seq):
        for sample in seq:
            self.module.activate(sample[0], sample[1])

        error = 0
        ponderation = 0.
        for offset, sample in reversed(list(enumerate(seq))):
            target = sample[1]
            outerr = target - self.module.activate(sample[0], sample[1])
            #print 0.5 * sum(outerr ** 2)
            error += 0.5 * sum(outerr ** 2)
            ponderation += len(target)
            self.module.back_activate()
        return error, ponderation
