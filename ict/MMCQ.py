import numpy as np
from queue import PriorityQueue as PQueue
from functools import reduce

# PQueue lowest first!
DEBUG = False

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
        self.plane    = ziped[:self.mAxis] + ziped[self.mAxis+1:]
        self.npixs    = self.population()
        self.priority = self.npixs * -1
    def population(self):
        s = 0
        for r in range(self.r1, self.r2+1):
            for g in range(self.g1, self.g2+1):
                for b in range(self.b1, self.b2+1):
                    s += self.histo[MMCQ.getColorIndex(r, g, b)]
        return int(s)
    def __lt__(self, vbox):    #实现<操作
        return self.priority < vbox.priority
    def contains(self, r, g, b):
        # real r, g, b here
        pass

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

        if DEBUG:
            print("Red range: {0}-{1}".format(rmin, rmax))
            print("Green range: {0}-{1}".format(gmin, gmax))
            print("Blue range: {0}-{1}".format(bmin, bmax))
        return VBox(rmin, rmax, gmin, gmax, bmin, bmax,self.pixHisto)
    def medianCutApply(self, vbox):
        npixs = 0
        if vbox.mAxis == 0:
            # Red axis is largest
            plane = 0
            for r in range(vbox.r1, vbox.r2+1):
                for g in range(vbox.g1, vbox.g2+1):
                    for b in range(vbox.b1, vbox.b2+1):
                        h = vbox.histo[self.getColorIndex(r, g, b)]
                        plane += h
                        npixs += h
                if npixs >= vbox.npixs / 2.:
                    left = r - vbox.r1
                    right = vbox.r2 - r
                    if left >= right:
                        r2 = int(max(vbox.r1, r - 1 - left / 2))
                    else:
                        r2 = int(min(vbox.r2 - 1, r + right / 2))
                    vbox1 = VBox(vbox.r1, r2, vbox.g1, vbox.g2, vbox.b1, vbox.b2, vbox.histo)
                    vbox2 = VBox(r2+1, vbox.r2, vbox.g1, vbox.g2, vbox.b1, vbox.b2, vbox.histo)
                    return vbox1, vbox2
        elif vbox.mAxis == 1:
            # Green axis is largest
            for g in range(vbox.g1, vbox.g2+1):
                plane = 0
                for r in range(vbox.r1, vbox.r2+1):
                    for b in range(vbox.b1, vbox.b2+1):
                        h = vbox.histo[self.getColorIndex(r, g, b)]
                        plane += h
                        npixs += h
                if npixs >= vbox.npixs / 2.:
                    left = g - vbox.g1
                    right = vbox.g2 - g
                    if left >= right:
                        g2 = int(max(vbox.g1, g - 1 - left / 2))
                    else:
                        g2 = int(min(vbox.g2 - 1, g + right / 2))
                    vbox1 = VBox(vbox.r1, vbox.r2, vbox.g1, g2, vbox.b1, vbox.b2, vbox.histo)
                    vbox2 = VBox(vbox.r1, vbox.r2, g2+1, vbox.g2, vbox.b1, vbox.b2, vbox.histo)
                    return vbox1, vbox2
        else:
            # Blue axis is largest
            for b in range(vbox.b1, vbox.b2+1):
                plane = 0
                for r in range(vbox.r1, vbox.r2+1):
                    for g in range(vbox.b1, vbox.b2+1):
                        h = vbox.histo[self.getColorIndex(r, g, b)]
                        plane += h
                        npixs += h
                if npixs >= vbox.npixs / 2.:
                    left = b - vbox.b1
                    right = vbox.b2 - b
                    if left >= right:
                        b2 = int(max(vbox.b1, b - 1 - left / 2))
                    else:
                        b2 = int(min(vbox.b2 - 1, b + right / 2))
                    vbox1 = VBox(vbox.r1, vbox.r2, vbox.g1, vbox.g2, vbox.b1, b2, vbox.histo)
                    vbox2 = VBox(vbox.r1, vbox.r2, vbox.g1, vbox.g2, b2+1, vbox.b2, vbox.histo)
                    return vbox1, vbox2
    def iterCut(self, maxColor, boxQueue, vol=False):
        ncolors = 1
        niters  = 0
        while True:
            if ncolors >= maxColor:
                break
            vbox0 = boxQueue.get_nowait()[1]
            if vbox0.npixs == 0:
                print("Vbox has no pixels")
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
            niters += 1
            if niters >= self.MAX_ITERATIONS:
                print("infinite loop; perhaps too few pixels!")
                break
        return boxQueue
    def boxAvgColor(self, vbox):
        ntot = 0
        mult = 1 << self.rshift
        rsum = 0
        gsum = 0
        bsum = 0
        for r in range(vbox.r1, vbox.r2+1):
            for g in range(vbox.g1, vbox.g2+1):
                for b in range(vbox.b1, vbox.b2+1):
                    h = vbox.histo[self.getColorIndex(r, g, b)]
                    ntot += h
                    rsum += int(h * (r + 0.5) * mult)
                    gsum += int(h * (g + 0.5) * mult)
                    bsum += int(h * (b + 0.5) * mult)
        if ntot == 0:
            avgs = map(lambda x: x * mult / 2, [vbox.r1 + vbox.r2 + 1, vbox.g1 + vbox.g2 + 1, vbox.b1 + vbox.b2 + 1])
        else:
            avgs = map(lambda x : x / ntot, [rsum, gsum, bsum])
        return list(map(lambda x: int(x), avgs))

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
            vbox = pOneQueue.get()[1]
            vbox.priority *= vbox.vol
            boxQueue.put((vbox.priority, vbox))
        boxQueue = self.iterCut(self.maxColor - popcolors + 1, boxQueue, True)

        theme = []
        while not boxQueue.empty():
            theme.append(self.boxAvgColor(boxQueue.get()[1]))
        return theme
