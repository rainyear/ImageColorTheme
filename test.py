#!../venv3/bin/python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

import time

from ict.MMCQ import MMCQ
# from ict.OQ import OQ

def imgPixInColorSpace(pixData):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3)

    im = fig.add_subplot(gs[0,0])
    im.imshow(pixData)
    im.set_title("2D Image")

    ax = fig.add_subplot(gs[0,1:3], projection='3d')
    colors = np.reshape(pixData, (pixData.shape[0] * pixData.shape[1], pixData.shape[2]))
    colors = colors / 255.0
    ax.scatter(pixData[:,:,0], pixData[:,:,1], pixData[:,:,2], c=colors)
    ax.set_xlabel("Red", color='red')
    ax.set_ylabel("Green", color='green')
    ax.set_zlabel("Blue", color='blue')

    ax.set_title("Image in Color Space")

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])


    plt.show()

def imgPalette(imgs, themes):
    N = len(imgs)

    fig = plt.figure()
    gs  = gridspec.GridSpec(len(imgs), 2)
    print(N)
    for i in range(N):
        theme = themes[i]
        pale = np.zeros(imgs[i].shape, dtype=np.uint8)
        h, w, _ = pale.shape
        ph  = h / len(theme)
        for y in range(h):
            pale[y,:,:] = np.array(theme[int(y / ph)], dtype=np.uint8)
        im = fig.add_subplot(gs[i, 0])
        im.imshow(imgs[i])
        im.set_title("Image %s" % str(i+1))
        im.xaxis.set_ticks([])
        im.yaxis.set_ticks([])

        pl = fig.add_subplot(gs[i, 1])
        pl.imshow(pale)
        pl.set_title("MMCQ Palette")
        pl.xaxis.set_ticks([])
        pl.yaxis.set_ticks([])

    plt.show()


if __name__ == '__main__':
    # imgfile = 'imgs/avatar_282x282.png'
    # pixData = cv.imread(imgfile, 1)
    # pixData = cv.cvtColor(pixData, cv.COLOR_BGR2RGB)
    # imgPixInColorSpace(cv.resize(pixData, None, fx=0.2, fy=0.2))

    imgs = map(lambda i: 'imgs/photo%s.jpg' % i, range(1,5))
    pixDatas = list(map(lambda f: cv.cvtColor(cv.imread(f, 1), cv.COLOR_BGR2RGB), imgs))
    maxColor = 7

    start  = time.process_time()
    themes = list(map(lambda d: MMCQ(d, maxColor).quantize(), pixDatas))
    print("Time cost: {0}".format(time.process_time() - start))
    imgPalette(pixDatas, themes)
