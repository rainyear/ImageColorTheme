import numpy as np
from queue import PriorityQueue as PQueue
from functools import reduce

# PQueue lowest first!

class VBox(object):
    """
        The color space is divided up into a set of 3D rectangular regions (called `vboxes`)
    """
    def __init__(self, r1, r2, g1, g2, b1, b2, histo):
        super(VBox, self).__init__()
        self.r1 = r1
        self.r2 = r2
        self.g1 = g1
        self.g2 = g2
        self.b1 = b1
        self.b2 = b2
        self.histo = histo

        ziped         = [(r1, r2), (g1, g2), (b1, b2)]
        sides         = list(map(lambda t: abs(t[0] - t[1]) + 1, ziped))
        self.vol      = reduce(lambda x, y: x*y, sides)
        self.mAxis    = sides.index(max(sides))
        self.plane    = ziped[:].pop(ziped[self.mAxis])
        self.npixs    = self.population()
        self.priority = self.npixs * -1
    def population(self):
        s = 0
        for r in range(self.r1, self.r2+1):
            for g in range(self.g1, self.g2+1):
                for b in range(self.b1, self.b2+1):
                    s += self.histo[MMCQ.getColorIndex(r, g, b)]
        return int(s)

class MMCQ(object):
    """
        Modified Median Cut Quantization(MMCQ)
        Leptonica: http://tpgit.github.io/UnOfficialLeptDocs/leptonica/color-quantization.html
    """
    MAX_ITERATIONS = 1000
    SIGBITS        = 5
    def __init__(self, pixData, maxColor, fraction=0.85, sigbits=5):
        """
        @pixData        Image data [[R, G, B], ...]
        @maxColor       Between [2, 256]
        @fraction       Between [0.3, 0.9]
        @sigbits        5 or 6
        """
        super(MMCQ, self).__init__()
        self.pixData  = pixData
        if not 2 <= maxColor <= 256:
            raise AttributeError("maxColor should between [2, 256]!")
        self.maxColor = maxColor
        if not 0.3 <= fraction <= 0.9:
            raise AttributeError("fraction should between [0.3, 0.9]!")
        self.fraction = fraction
        if sigbits != 5 and sigbits != 6:
            raise AttributeError("sigbits should be either 5 or 6!")
        self.SIGBITS = sigbits
        self.rshift  = 8 - sigbits

        self.h, self.w, _ = self.pixData.shape
    def getPixHisto(self):
        pixHisto = np.zeros(1 << (3 * self.SIGBITS))
        for y in range(self.h):
            for x in range(self.w):
                r = self.pixData[y, x, 0] >> self.rshift
                g = self.pixData[y, x, 1] >> self.rshift
                b = self.pixData[y, x, 2] >> self.rshift

                pixHisto[self.getColorIndex(r, g, b)] += 1
        return pixHisto
    @classmethod
    def getColorIndex(self, r, g, b):
        return (r << (2 * self.SIGBITS)) + (g << self.SIGBITS) + b
    def createVbox(self, pixData):
        rmax = np.max(pixData[:,:,0]) >> self.rshift
        rmin = np.min(pixData[:,:,0]) >> self.rshift
        gmax = np.max(pixData[:,:,1]) >> self.rshift
        gmin = np.min(pixData[:,:,1]) >> self.rshift
        bmax = np.max(pixData[:,:,2]) >> self.rshift
        bmin = np.min(pixData[:,:,2]) >> self.rshift

        print("Red range: {0}-{1}".format(rmin, rmax))
        print("Green range: {0}-{1}".format(gmin, gmax))
        print("Blue range: {0}-{1}".format(bmin, bmax))
        return VBox(rmin, rmax, gmin, gmax, bmin, bmax,self.pixHisto)
    def medianCutApply(self, vbox):
        if vbox.mAxis == 0:
            # Red axis is largest
            pass
        elif vbox.mAxis == 1:
            # Green axis is largest
            pass
        else:
            # Blue axis is largest
            pass
    def iterCut(self, maxColor, boxQueue, vol=False):
        ncolors = 1
        niters  = 0
        while True:
            vbox0 = boxQueue.get()
            if vbox0.npixs == 0:
                boxQueue.put((vbox0.priority, vbox0))
                continue
            vbox1, vbox2 = self.medianCutApply(vbox0)

            if vol:
                vbox1.priority *= vbox1.vol
            boxQueue.put((vbox1.priority, vbox1))
            if vbox2 is not None:
                ncolors += 1
                if vol:
                    vbox2.priority *= vbox2.vol
                boxQueue.put((vbox2.priority, vbox2))
            if ncolors >= maxColor:
                break
            niters += 1
            if niters >= self.MAX_ITERATIONS:
                print("infinite loop; perhaps too few pixels!")
                break
        return boxQueue
    def quantize(self):
        if self.h * self.w < self.maxColor:
            raise AttributeError("Image({0}x{1}) too small to be quantized".format(self.w, self.h))
        self.pixHisto = self.getPixHisto()

        orgVbox = self.createVbox(self.pixData)
        pOneQueue = PQueue(self.maxColor)
        pOneQueue.put((orgVbox.priority, orgVbox))
        popcolors = int(self.maxColor * self.fraction)

        pOneQueue = self.iterCut(popcolors, pOneQueue)

        boxQueue = PQueue(self.maxColor)
        while not pOneQueue.empty():
            vbox = pOneQueue.get()
            vbox.priority *= vbox.vol
            boxQueue.put((vbox.priority, vbox))
        boxQueue = self.iterCut(self.maxColor - popcolors, boxQueue, True)
