# This is the projection code for CityStreet dataset.
#
# ---
# CityStreet: Multi-View Crowd Counting Dataset
# Copyright (c) 2019
# Qi Zhang, Antoni B. Chan
# City University of Hong Kong

import json
import cv2
# from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
import scipy.io as sio
# from skimage.transform import downscale_local_mean
# import skimage.io
import os
import sys
import math
import time
import random
from random import shuffle
import pickle
import h5py
import glob


def view1ToView2(u, v, Zw):
    XYw = Image2World("view1", u, v, Zw)
    XYi2 = World2Image("view2", XYw[0], XYw[1], Zw)
    return XYi2


def view1ToView3(u, v, Zw):
    XYw = Image2World("view1", u, v, Zw)
    XYi3 = World2Image("view3", XYw[0], XYw[1], Zw)
    return XYi3


def view2ToView3(u, v, Zw):
    XYw = Image2World("view2", u, v, Zw)
    XYi3 = World2Image("view3", XYw[0], XYw[1], Zw)
    return XYi3


def view2ToView1(u, v, Zw):
    XYw = Image2World("view2", u, v, Zw)
    XYi3 = World2Image("view1", XYw[0], XYw[1], Zw)
    return XYi3


def view3ToView1(u, v, Zw):
    XYw = Image2World("view3", u, v, Zw)
    XYi3 = World2Image("view1", XYw[0], XYw[1], Zw)
    return XYi3


def view3ToView2(u, v, Zw):
    XYw = Image2World("view3", u, v, Zw)
    XYi3 = World2Image("view2", XYw[0], XYw[1], Zw)
    return XYi3


def find_height(x1, y1, x2, y2):
    height_range = range(1560, 1980, 5)
    dist_error = np.asarray(height_range.shape)
    index = 0
    error_num = 0

    for i in range(height_range.shape[0]):
        height_i = height_range[i]
        g1 = Image2World("view1", x1, y1, height_i)
        g2 = Image2World("view2", x2, y2, height_i)

        gx = g1[0] - g2[0]
        gy = g1[1] - g2[1]

        error_i = np.sqrt(gx * gx + gy * gy)
        if (i == 0):
            index = 0
            error_num = error_i
        else:
            if (error_num > error_i):
                error_num = error_i
                index = i

    height = height_range[index]
    g3 = view1ToView3(x1, y1, height)

    return [g3[0], g3[1], height]


def find_height2(x2, y2, x3, y3):
    height_range = range(1560, 1980, 5)
    dist_error = np.asarray(height_range.shape)
    index = 0
    error_num = 0

    for i in range(height_range.shape[0]):
        height_i = height_range[i]
        g2 = Image2World("view2", x2, y2, height_i)
        g3 = Image2World("view3", x3, y3, height_i)

        gx = g2[0] - g3[0]
        gy = g2[1] - g3[1]

        error_i = np.sqrt(gx * gx + gy * gy)
        if (i == 0):
            index = 0
            error_num = error_i
        else:
            if (error_num > error_i):
                error_num = error_i
                index = i

    height = height_range[index]

    return height


def find_height_3views(x1, y1, x2, y2, x3, y3):
    height_range = range(1560, 1980, 5)
    dist_error = np.asarray(height_range.shape)

    index = 0
    error_num = 0

    for i in range(height_range.shape[0]):

        height_i = height_range[i]
        g1 = Image2World("view1", x1, y1, height_i)
        g2 = Image2World("view2", x2, y2, height_i)
        g3 = Image2World("view3", x3, y3, height_i)

        gx_avg = (g1[0] + g2[0] + g3[0]) / 3.0
        gy_avg = (g1[1] + g2[1] + g3[1]) / 3.0

        gx1 = g1[0] - gx_avg
        gy1 = g1[1] - gy_avg
        gx2 = g2[0] - gx_avg
        gy2 = g2[1] - gy_avg
        gx3 = g3[0] - gx_avg
        gy3 = g3[1] - gy_avg

        error_i = np.sqrt(gx1 * gx1 + gy1 * gy1 + gx2 * gx2 + gy2 * gy2 + gx3 * gx3 + gy3 * gy3)

        if (i == 0):
            index = 0
            error_num = error_i
        else:
            if (error_num > error_i):
                error_num = error_i
                index = i

    height = height_range[index]

    return height


def camerafile(view):
    camerainfo = {}
    if view == 'view1':
        camerainfo['width'] = 2704
        camerainfo['height'] = 1520
        camerainfo['ncx'] = 7.9500000000e+02
        camerainfo['nfx'] = 7.5200000000e+02

        camerainfo['dx'] = 4.8500000000e-03
        camerainfo['dy'] = 4.6500000000e-03
        camerainfo['dpx'] = 1 / (1.2434e+03)
        camerainfo['dpy'] = 1 / (1.2489e+03)

        camerainfo['focal'] = 1
        camerainfo['kappa1'] = -0.2090
        camerainfo['kappa2'] = 0.0382

        camerainfo['cx'] = 1.38455e+3
        camerainfo['cy'] = 747.846499320945
        camerainfo['sx'] = 1

        camerainfo['tx'] = 1275.94012233
        camerainfo['ty'] = 226.52473842
        camerainfo['tz'] = 24230.24027107

        camerainfo['rx'] = 2.0405458695e+00
        camerainfo['ry'] = -8.9337703748e-01
        camerainfo['rz'] = -4.3056124791e-01

    if view == 'view2':
        camerainfo['width'] = 2704
        camerainfo['height'] = 1520
        camerainfo['ncx'] = 7.9500000000e+02
        camerainfo['nfx'] = 7.5200000000e+02

        camerainfo['dx'] = 4.8500000000e-03
        camerainfo['dy'] = 4.6500000000e-03
        camerainfo['dpx'] = 1 / (1.25818922e+03)
        camerainfo['dpy'] = 1 / (1.25988567e+03)

        camerainfo['focal'] = 1
        camerainfo['kappa1'] = -0.23977946
        camerainfo['kappa2'] = 0.05901853

        camerainfo['cx'] = 1.29253760e+03
        camerainfo['cy'] = 7.70518094e+02
        camerainfo['sx'] = 1

        camerainfo['tx'] = 9680.47673912
        camerainfo['ty'] = 2273.39720387
        camerainfo['tz'] = 31436.76306171

        camerainfo['rx'] = 1.7637554450e+00
        camerainfo['ry'] = -6.8268312644e-01
        camerainfo['rz'] = -6.3321351894e-02

    if view == 'view3':
        camerainfo['width'] = 2704
        camerainfo['height'] = 1520
        camerainfo['ncx'] = 7.9500000000e+02
        camerainfo['nfx'] = 7.5200000000e+02

        camerainfo['dx'] = 4.8500000000e-03
        camerainfo['dy'] = 4.6500000000e-03
        camerainfo['dpx'] = 1 / (1255.9)
        camerainfo['dpy'] = 1 / (1257.2)

        camerainfo['focal'] = 1
        camerainfo['kappa1'] = -0.2489
        camerainfo['kappa2'] = 0.07147814

        camerainfo['cx'] = 1.3741e+3
        camerainfo['cy'] = 785.7148
        camerainfo['sx'] = 1

        camerainfo['tx'] = 3309.55016035
        camerainfo['ty'] = -4603.90350916
        camerainfo['tz'] = 29264.19379322

        camerainfo['rx'] = 1.8665618542e+00
        camerainfo['ry'] = 1.5219705811e-01
        camerainfo['rz'] = 4.5968889283e-02
    return camerainfo


def Image2World(view, Xi, Yi, Zw):
    Xi = Xi
    Yi = Yi
    Zw = -Zw

    if view == "view1":
        camerainfo = camerafile("view1");
        mR11 = -0.03615359
        mR12 = 0.99596323
        mR13 = -0.0821594
        mR21 = -0.34251474
        mR22 = 0.06488427
        mR23 = 0.93726927
        mR31 = 0.93881658
        mR32 = 0.06202646
        mR33 = 0.33878628

    if view == "view2":
        camerainfo = camerafile("view2")
        mR11 = -0.99730688
        mR12 = 0.06324747
        mR13 = 0.03713147
        mR21 = 0.02439897
        mR22 = -0.19132768
        mR23 = 0.9812229
        mR31 = 0.06916415
        mR32 = 0.97948633
        mR33 = 0.18926923

    if view == "view3":
        camerainfo = camerafile("view3")
        mR11 = 0.99545502
        mR12 = 0.07578782
        mR13 = -0.05766726
        mR21 = 0.0167508
        mR22 = 0.45675628
        mR23 = 0.88943415
        mR31 = 0.09374816
        mR32 = -0.88635766
        mR33 = 0.45341083

    mDx = camerainfo['dx']
    mDy = camerainfo['dy']
    mDpx = camerainfo['dpx']
    mDpy = camerainfo['dpy']
    mFocal = camerainfo['focal']
    mKappa1 = camerainfo['kappa1']
    mKappa2 = camerainfo['kappa2']
    mCx = camerainfo['cx']
    mCy = camerainfo['cy']
    mSx = camerainfo['sx']
    mTx = camerainfo['tx']
    mTy = camerainfo['ty']
    mTz = camerainfo['tz']
    mRx = camerainfo['rx']
    mRy = camerainfo['ry']
    mRz = camerainfo['rz']

    # world coords image coords, rotation and transition
    sa = np.sin(mRx)
    ca = np.cos(mRx)
    sb = np.sin(mRy)
    cb = np.cos(mRy)
    sg = np.sin(mRz)
    cg = np.cos(mRz)

    # mR11 = cb * cg
    # mR12 = cg * sa * sb - ca * sg
    # mR13 = sa * sg + ca * cg * sb
    # mR21 = cb * sg
    # mR22 = sa * sb * sg + ca * cg
    # mR23 = ca * sb * sg - cg * sa
    # mR31 = -sb
    # mR32 = cb * sa
    # mR33 = ca * cb

    # compute camera position
    mCposx = -(mTx * mR11 + mTy * mR21 + mTz * mR31)
    mCposy = -(mTx * mR12 + mTy * mR22 + mTz * mR32)
    mCposz = -(mTx * mR13 + mTy * mR23 + mTz * mR33)

    # step 1:
    Xd = mDpx * (Xi - mCx) / mSx
    Yd = mDpy * (Yi - mCy)

    # step 2:
    distortion_factor = 1 + mKappa1 * (Xd * Xd + Yd * Yd)
    Xu = Xd * distortion_factor
    Yu = Yd * distortion_factor

    # step 3:
    common_denominator = ((mR11 * mR32 - mR12 * mR31) * Yu +
                          (mR22 * mR31 - mR21 * mR32) * Xu -
                          mFocal * mR11 * mR22 + mFocal * mR12 * mR21)

    Xw = (((mR12 * mR33 - mR13 * mR32) * Yu +
           (mR23 * mR32 - mR22 * mR33) * Xu -
           mFocal * mR12 * mR23 + mFocal * mR13 * mR22) * Zw +
          (mR12 * mTz - mR32 * mTx) * Yu +
          (mR32 * mTy - mR22 * mTz) * Xu -
          mFocal * mR12 * mTy + mFocal * mR22 * mTx) / common_denominator

    Yw = -(((mR11 * mR33 - mR13 * mR31) * Yu +
            (mR23 * mR31 - mR21 * mR33) * Xu -
            mFocal * mR11 * mR23 + mFocal * mR13 * mR21) * Zw +
           (mR11 * mTz - mR31 * mTx) * Yu +
           (mR31 * mTy - mR21 * mTz) * Xu -
           mFocal * mR11 * mTy + mFocal * mR21 * mTx) / common_denominator

    return [Xw, Yw, -Zw]


def World2Image(view, Xw, Yw, Zw):
    Zw = -Zw

    if view == "view1":
        camerainfo = camerafile("view1");
        mR11 = -0.03615359
        mR12 = 0.99596323
        mR13 = -0.0821594
        mR21 = -0.34251474
        mR22 = 0.06488427
        mR23 = 0.93726927
        mR31 = 0.93881658
        mR32 = 0.06202646
        mR33 = 0.33878628

    if view == "view2":
        camerainfo = camerafile("view2")
        mR11 = -0.99730688
        mR12 = 0.06324747
        mR13 = 0.03713147
        mR21 = 0.02439897
        mR22 = -0.19132768
        mR23 = 0.9812229
        mR31 = 0.06916415
        mR32 = 0.97948633
        mR33 = 0.18926923

    if view == "view3":
        camerainfo = camerafile("view3")
        mR11 = 0.99545502
        mR12 = 0.07578782
        mR13 = -0.05766726
        mR21 = 0.0167508
        mR22 = 0.45675628
        mR23 = 0.88943415
        mR31 = 0.09374816
        mR32 = -0.88635766
        mR33 = 0.45341083

    mDx = camerainfo['dx']
    mDy = camerainfo['dy']
    mDpx = camerainfo['dpx']
    mDpy = camerainfo['dpy']
    mFocal = camerainfo['focal']
    mKappa1 = camerainfo['kappa1']
    mKappa2 = camerainfo['kappa2']
    mCx = camerainfo['cx']
    mCy = camerainfo['cy']
    mSx = camerainfo['sx']
    mTx = camerainfo['tx']
    mTy = camerainfo['ty']
    mTz = camerainfo['tz']
    mRx = camerainfo['rx']
    mRy = camerainfo['ry']
    mRz = camerainfo['rz']

    # world coordsimage coords, rotation and transition
    sa = np.sin(mRx)
    ca = np.cos(mRx)
    sb = np.sin(mRy)
    cb = np.cos(mRy)
    sg = np.sin(mRz)
    cg = np.cos(mRz)

    # mR11 = cb * cg
    # mR12 = cg * sa * sb - ca * sg
    # mR13 = sa * sg + ca * cg * sb
    # mR21 = cb * sg
    # mR22 = sa * sb * sg + ca * cg
    # mR23 = ca * sb * sg - cg * sa
    # mR31 = -sb
    # mR32 = cb * sa
    # mR33 = ca * cb

    # compute camera position
    mCposx = -(mTx * mR11 + mTy * mR21 + mTz * mR31)
    mCposy = -(mTx * mR12 + mTy * mR22 + mTz * mR32)
    mCposz = -(mTx * mR13 + mTy * mR23 + mTz * mR33)

    # step 1:
    xc = mR11 * Xw + mR12 * Yw + mR13 * Zw + mTx
    yc = mR21 * Xw + mR22 * Yw + mR23 * Zw + mTy
    zc = mR31 * Xw + mR32 * Yw + mR33 * Zw + mTz

    # step 2:
    Xu = mFocal * xc / zc
    Yu = mFocal * yc / zc

    # step 3:
    XYd = undistortedToDistortedSensorCoord(Xu, Yu, mKappa1, mKappa2)
    # XYd = undistortedToDistortedSensorCoord(Xu, Yu, mKappa1) # 2022-07-26改
    Xd = XYd[0]
    Yd = XYd[1]

    # step 4:
    Xi = Xd * mSx / mDpx + mCx
    Yi = Yd / mDpy + mCy
    return [Xi, Yi, -Zw]


def undistortedToDistortedSensorCoord(Xu, Yu, mKappa1, mKappa2):
    k = 0
    e = 1e-10

    if (((Xu == 0) & (Yu == 0)) | (mKappa1 == 0) & (mKappa2 == 0)):
        Xd = Xu
        Yd = Yu
    else:
        Ru = np.sqrt(Xu * Xu + Yu * Yu)
        a = mKappa2 * mKappa2
        b = 2 * mKappa1 * mKappa2
        c = mKappa1 * mKappa1 + 2 * mKappa2
        d = 2 * mKappa1
        Rd = Ru

        while (k < 1000):
            k = k + 1
            R0 = Rd

            fR0 = a * np.power(R0, 5) + b * np.power(R0, 4) + c * np.power(R0, 3) + d * np.power(R0, 2) + R0 - Ru
            f_R0 = 5 * a * np.power(R0, 4) + 4 * b * np.power(R0, 3) + 3 * c * R0 * R0 + 2 * d * R0 + 1

            Rd = R0 - fR0 / f_R0
            if (np.abs(Rd - R0) <= e):
                break

        lambda0 = 1 + mKappa1 * Rd + mKappa2 * Rd * Rd
        Xd = Xu / lambda0
        Yd = Yu / lambda0
    return [Xd, Yd]


def undistortedToDistortedSensorCoord0(Xu, Yu, mKappa1):
    if (((Xu == 0) & (Yu == 0)) | (mKappa1 == 0)):
        Xd = Xu
        Yd = Yu
    else:
        Ru = np.sqrt(Xu * Xu + Yu * Yu)
        c = 1.0 / mKappa1
        d = -c * Ru
        Q = c / 3
        R = -d / 2
        D = Q * Q * Q + R * R
        if (D >= 0):
            # /// * one real root */
            D = np.sqrt(D)
            if (R + D > 0):
                S = np.power(R + D, 1.0 / 3.0)
            else:
                S = -np.power(-R - D, 1.0 / 3.0)

            if (R - D > 0):
                T = np.power(R - D, 1.0 / 3.0)
            else:
                T = -np.power(D - R, 1.0 / 3.0)

            Rd = S + T

            if (Rd < 0):
                Rd = np.sqrt(-1.0 / (3 * mKappa1))

        else:
            # /* three real roots */
            D = np.sqrt(-D)
            S = np.power(np.sqrt(R * R + D * D), 1.0 / 3.0)
            T = np.arctan2(D, R) / 3
            sinT = np.sin(T)
            cosT = np.cos(T)
            # /* the larger positive root is    2*S*cos(T)                   */
            # /* the smaller positive root is   -S*cos(T) + SQRT(3)*S*sin(T) */
            # /* the negative root is           -S*cos(T) - SQRT(3)*S*sin(T) */
            Rd = -S * cosT + np.sqrt(3.0) * S * sinT  # /* use the smaller positive root */

        lambda0 = Rd / Ru
        Xd = Xu * lambda0
        Yd = Yu * lambda0
    return [Xd, Yd]

if __name__ == '__main__':
    pass