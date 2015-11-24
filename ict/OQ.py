import numpy as np

class OctNode(object):
    """docstring for OctNode"""
    isLeaf = False
    n      = 0
    r      = 0
    g      = 0
    b      = 0
    next   = None
    def __init__(self):
        super(OctNode, self).__init__()
        self.children = []
        for i in range(8):
            self.children.append(None)

class OQ(object):
    """Octree Quantization"""
    NCOLORS = 10
    def __init__(self, pixData, maxColor):
        super(OQ, self).__init__()
        self.pixData = pixData
        if not 2 <= maxColor <= 256:
            raise AttributeError("maxColor should be [2, 256]")

        self.maxColor     = maxColor
        self.H, self.W, _ = self.pixData.shape
        self.leafNum      = 0
        self.reducible    = []
        self.theme        = []

        for i in range(7):
            self.reducible.append(None)

        self.octree = OctNode()
    def buildOctree(self):
        for y in range(self.H):
            for x in range(self.W):
                pix = self.pixData[y, x, :]
                self.addColor(self.octree, pix, 0)
    def addColor(self, node, pix, level):
        if node.isLeaf:
            node.n += 1
            node.r += pix[0]
            node.g += pix[1]
            node.b += pix[2]
        else:
            rc = ((pix[0] >> (7 - level)) & 0x1) << 2
            gc = ((pix[1] >> (7 - level)) & 0x1) << 1
            bc = (pix[2] >> (7 - level)) & 0x1
            idx = rc | gc | bc

            if node.children[idx] is None:
                node.children[idx] = self.createOctNode(level + 1)
            self.addColor(node.children[idx], pix, level + 1)
    def createOctNode(self, level):
        node = OctNode()
        if level == 7:
            node.isLeaf   = True
            self.leafNum += 1
        else:
            node.next             = self.reducible[level]
            self.reducible[level] = node
        return node
    def reduceTree(self):
        lv = 6
        while self.reducible[lv] is None:
            lv -= 1
        node = self.reducible[lv]
        self.reducible[lv] = node.next

        r, g, b, c = (0, 0, 0, 0)
        for i in range(8):
            child = node.children[i]
            if child is None:
                continue
            r += child.r
            g += child.g
            b += child.b
            c += child.n
            self.leafNum -= 1

        node.isLeaf   = True
        node.r        = r
        node.g        = g
        node.b        = b
        node.n       += c
        self.leafNum += 1
    def getColors(self, node):
        if node.isLeaf:
            [r, g, b] = list(map(lambda n: int(n[0] / n[1]), zip([node.r, node.g, node.b], [node.n]*3)))
            self.theme.append([r,g,b, node.n])
        else:
            for i in range(8):
                if node.children[i] is not None:
                    self.getColors(node.children[i])
    def quantize(self):
        self.buildOctree()
        if self.leafNum <= self.maxColor:
            raise AttributeError("Image too small to be quantized!")
        while self.leafNum > (self.maxColor + self.NCOLORS):
            self.reduceTree()
            # print("leafNum = {0}".format(self.leafNum))
        self.getColors(self.octree)
        # print(len(self.theme))
        self.theme = sorted(self.theme, key=lambda c: -1*c[1])
        return list(map(lambda l: l[:-1],self.theme[:self.maxColor]))
