#!/home/utils/Python-3.5.2/bin/python3

import os
import struct
import numpy as np
import cv2


def disk(radius):
    diameter = 1 + radius * 2
    pattern = np.ones((diameter, diameter), np.uint8)
    for y in range(diameter):
        for x in range(diameter):
            if (y - radius) ** 2 + (x - radius) ** 2 > radius ** 2:
                pattern[y, x] = 0
    return pattern


def renderErrImg(err, valid, *, dilateRadius=1):
    colorMap = (
        ( 0     /3.0,  0.1875/3.0,  49,  54, 149),
        ( 0.1875/3.0,  0.375 /3.0,  69, 117, 180),
        ( 0.375 /3.0,  0.75  /3.0, 116, 173, 209),
        ( 0.75  /3.0,  1.5   /3.0, 171, 217, 233),
        ( 1.5   /3.0,  3     /3.0, 224, 243, 248),
        ( 3     /3.0,  6     /3.0, 254, 224, 144),
        ( 6     /3.0, 12     /3.0, 253, 174,  97),
        (12     /3.0, 24     /3.0, 244, 109,  67),
        (24     /3.0, 48     /3.0, 215,  48,  39),
        (48     /3.0,      np.inf, 165,   0,  38),
    )
    assert err.ndim == 2
    size = err.shape
    r = np.zeros(size, np.uint8)
    g = np.zeros(size, np.uint8)
    b = np.zeros(size, np.uint8)
    for cmap in colorMap:
        mask = valid & (err >= cmap[0]) & (err < cmap[1])
        r[mask] = cmap[2]
        g[mask] = cmap[3]
        b[mask] = cmap[4]
    img = cv2.merge([b, g, r])
    if dilateRadius > 0:
        img = cv2.dilate(img, disk(dilateRadius))
    return img



def readDispFromPng(filepath):
    """Read disparity from .png in KITTI format
       To support other format, a mode parameter might be added.
    """
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    disp = img / np.float32(256.0)
    invalids = img == 0
    disp[invalids] = -1 # set invalids to -1 to avoid divede-by-0 trouble
    return disp


def readPfm(filepath):
    with open(filepath, 'rb') as f:
        magic = f.readline().strip().decode('ascii')
        assert magic == 'Pf'
        width, height = f.readline().strip().decode('ascii').split()
        width, height = int(width), int(height)
        scalefactor = float(f.readline().strip().decode('ascii'))
        littleEndian = scalefactor < 0
        fmt = '{}{}f'.format('<' if littleEndian else '>', width) 
        Bpp = 4
        img = np.zeros((height, width), dtype=np.float32)
        for y in range(height - 1, -1, -1): # pfm stores lines in reverse order
            buf = f.read(width * Bpp)
            unpacked = struct.unpack(fmt, buf)
            img[y, :] = unpacked
    return img


def readDispFromPfm(filepath):
    """Read disparity from .pfm"""
    disp = readPfm(filepath)
    invalids = disp == np.inf
    disp[invalids] = -1
    return disp


def readDisp(filepath):
    if filepath.endswith('.png'):
        disp = readDispFromPng(filepath)
    elif filepath.endswith('.pfm'):
        disp = readDispFromPfm(filepath)
    else:
        assert False, 'no support to read disparity from {}'.format(os.path.basename(filepath))
    assert disp.dtype.type is np.float32
    return disp


def validDisp(disp):
    assert disp.ndim == 2
    return disp > 0


def renderDispImg(disp, *, maxDisp=-1):
    assert len(disp.shape) == 2
    colorMap = np.array([[0, 0, 0, 114], # smallest disparity
                         [0, 0, 1, 185],
                         [1, 0, 0, 114],
                         [1, 0, 1, 174],
                         [0, 1, 0, 114],
                         [0, 1, 1, 185],
                         [1, 1, 0, 114],
                         [1, 1, 1, 0]],  # largest disparity
                         dtype=np.float)
    bins = np.array([vec[3] for vec in colorMap[:-1]])
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]
    if maxDisp < 0:
        maxDisp = disp.max()
        print('maxDisp = {}'.format(maxDisp))
    disp = np.clip(disp / maxDisp, 0, 1.0)
    ind = np.digitize(disp, cbins, right=True)
    # print(np.histogram(ind, list(range(ind.max() + 1)))) # debug
    cbins = np.insert(cbins, 0, 0)
    w = (disp - cbins[ind]) / bins[ind] # w is smaller when disp is closer to cbins[ind]
    R = colorMap[:, 0][ind] * (1 - w) + colorMap[:, 0][ind + 1] * w
    G = colorMap[:, 1][ind] * (1 - w) + colorMap[:, 1][ind + 1] * w
    B = colorMap[:, 2][ind] * (1 - w) + colorMap[:, 2][ind + 1] * w
    return np.uint8(cv2.merge([B, G, R]) * 255)


def evalDispErr(estimated, *, gt):
    assert estimated.ndim == 2, 'only single plane'
    assert gt.ndim == 2, 'only single plane'
    gtValid = validDisp(gt) & validDisp(estimated)
    err = np.fabs(estimated - gt)
    err[~gtValid] = np.nan
    return err, gtValid

def evalDispErrMetric(disp, gt, tau=(3, 0.05)):
    valid = gt > 0
    err = np.fabs(disp - gt)
    bad = (err > tau[0]) & (err / np.fabs(gt) > tau[1])
    badPercentage = bad[valid].sum()/ valid[valid].sum()
    aepe = err[valid].mean()
    return aepe, badPercentage

def renderDispErrImg(est, *, gt, gtNoc=None, tau=(3, 0.05), dilateRadius=1):
    err = np.fabs(est - gt)
    err = np.fmin(err / tau[0], err / np.fabs(gt) / tau[1])
    img = renderErrImg(err, validDisp(gt) & validDisp(est), dilateRadius=0)
    if gtNoc is not None:
        mask = np.ones(img.shape, dtype=np.float32)
        mask[~validDisp(gtNoc)] = 0.5
        img = np.uint8(np.float32(img) * mask)
    if dilateRadius > 0:
        img = cv2.dilate(img, disk(dilateRadius))
    return img



VALID_PLANE = 0 # might be uncertainty plane
MVY_PLANE   = 1
MVX_PLANE   = 2


def readFlowFromPng(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    valid, g, r = cv2.split(img)
    mvy = (np.float32(g) - 32768) / 64
    mvx = (np.float32(r) - 32768) / 64
    return cv2.merge([np.float32(valid), mvy, mvx])


def readFlowFromFlo(filepath):
    with open(filepath, 'rb') as f:
        magic = f.read(4).decode('ascii')
        assert magic == 'PIEH'
        fmt = '<2i'
        buf = f.read(8)
        width, height = struct.unpack(fmt, buf)
        Bpp = 4
        nfloat = width * height * 2
        fmt = '<{}f'.format(nfloat)
        buf = f.read(nfloat * Bpp)
        flow = np.zeros((nfloat), dtype=np.float32)
        flow[:] = struct.unpack(fmt, buf)
        flow.shape = (height, width, 2)
    mvx, mvy = cv2.split(flow)
    valid = np.ones((height, width), dtype=np.float32)
    return cv2.merge([valid, mvy, mvx])


def readFlow(filepath):
    if filepath.endswith('.png'):
        return readFlowFromPng(filepath)
    elif filepath.endswith('.flo'):
        return readFlowFromFlo(filepath)
    else:
        assert False, 'no support to read flow from {}'.format(os.path.basename(filepath))


def validFlow(flow):
    if flow.ndim == 3:
        return flow[:,:,VALID_PLANE] == 1
    elif flow.ndim == 2:
        return np.ones_like(flow[:,:,VALID_PLANE])
    else:
        assert False, 'Wrong flow channel number'


def renderFlowImg(flow, *, maxFlow=-1, style='mpi'):
    assert style in ['kitti-c++', 'mpi'], 'unknown flow rendering style: {}'.format(style)
    assert len(flow.shape) == 3
    mvx = flow[:,:,MVX_PLANE]
    mvy = flow[:,:,MVY_PLANE]
    valid = validFlow(flow)
    magnitude = np.sqrt(mvx ** 2 + mvy ** 2)
    direction = np.arctan2(mvy, mvx)
    if maxFlow < 0:
        if style == 'kitti-c++':
            maxFlow = np.fmax(magnitude.max(), 1.0)
        else:
            # mpi style
            maxFlow = 1.2 * max(np.fabs(mvx).max(), np.fabs(mvy).max())
            maxFlow = np.fmax(maxFlow, 1.0)
        print('maxFlow = {:.1f}'.format(maxFlow))
    h = np.fmod(direction / (2 * np.pi) + 1.0, 1.0) * 360
    assert h.min() >= 0
    if style == 'kitti-c++':
        """KITTI C++ hsv2rgb has bug. Here tries to mimic the final result"""
        s = np.clip(magnitude * 300 / maxFlow, 0, 1.0)
        v = np.clip(magnitude * 8   / maxFlow, 0, 1.0)
    else:
        # mpi style
        n = 8
        s = np.clip(magnitude * n / maxFlow, 0, 1.0)
        v = np.clip(n - s, 0, 1.0)
    img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    img[~valid] = 0
    return np.uint8(img * 255)


def drawFlowVector(flow, *, step=8):
    scratch = np.zeros(flow.shape, dtype=np.uint8)
    for y in range(0, flow.shape[0], step):
        for x in range(0, flow.shape[1], step):
            x2 = x + np.int(np.rint(flow[y,x,MVX_PLANE]))
            y2 = y + np.int(np.rint(flow[y,x,MVY_PLANE]))
            color = (0, 192, 0)
            cv2.arrowedLine(scratch, (x, y), (x2, y2), color, thickness=1, line_type=cv2.LINE_AA)
    return scratch


def evalFlowErr(est, *, gt):
    dMvx = est[:,:,MVX_PLANE] - gt[:,:,MVX_PLANE]
    dMvy = est[:,:,MVY_PLANE] - gt[:,:,MVY_PLANE]
    err = np.sqrt(dMvx ** 2 + dMvy ** 2)
    valid = validFlow(est) & validFlow(gt)
    err[~valid] = 0
    return err, valid


# TODO refine, DO NOT use yet
def __evalErrMetric(err, valid, gt, tau=(3, 0.05)):
    magnitude = np.sqrt(gt[:,:,MVX_PLANE] ** 2 + gt[:,:,MVY_PLANE] ** 2)
    bad = valid & (err > tau[0]) & (err / magnitude > tau[1])
    badPercentage = bad.sum() / valid.sum()
    aepe = err.sum() / valid.sum()
    return aepe, badPercentage


def renderFlowErrImg(est, *, gt, gtNoc=None, tau=(3, 0.05), dilateRadius=1):
    dMvx = est[:,:,MVX_PLANE] - gt[:,:,MVX_PLANE]
    dMvy = est[:,:,MVY_PLANE] - gt[:,:,MVY_PLANE]
    err = np.sqrt(dMvx ** 2 + dMvy ** 2)
    estmag = np.sqrt(est[:,:,MVX_PLANE] ** 2 + est[:,:,MVY_PLANE] ** 2)
    err = np.fmin(err / tau[0], err / estmag / tau[1])
    valid = validFlow(est) & validFlow(gt)
    img = renderErrImg(err, valid, dilateRadius=0)
    if gtNoc is not None:
        mask = np.ones(img.shape, dtype=np.uint8)
        mask[~validFlow(gtNoc)] = 0.5
        img = np.uint8(np.float32(img) * mask)
    if dilateRadius > 0:
        img = cv2.dilate(img, disk(dilateRadius))
    return img

def writeFlowPng(filepath, flow):
    
    flow = flow*64 + 32768
    valid = np.ones((flow.shape[0], flow.shape[1], 1), dtype=np.float32)
    flow = np.concatenate((flow, valid), axis=2).astype(np.uint16)
    flow = flow[...,::-1].copy()
    cv2.imwrite(filepath, flow)

def writeDispPng(filepath, disp):
    disp = np.clip(256*disp, 0, 65535)
    disp = disp.astype(np.uint16)
    cv2.imwrite(filepath, disp)
