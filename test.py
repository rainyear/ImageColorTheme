#!../venv3/bin/python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec


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

if __name__ == '__main__':
    imgfile = 'avatar_282x282.png'
    pixData = cv.imread(imgfile, 1)
    pixData = cv.cvtColor(pixData, cv.COLOR_BGR2RGB)

    # imgPixInColorSpace(cv.resize(pixData, None, fx=0.2, fy=0.2))
    maxColor = 5
    mmcq = MMCQ(pixData, maxColor)
    mmcq.quantize()
