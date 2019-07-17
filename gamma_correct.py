# Copyright (c) 2019 kamino410. All rights reserved.
# This code is licensed under MIT license (see LICENSE.txt for details)

import sys
import os
import os.path
import re
import glob

import cv2
import numpy as np
from scipy.optimize import line_search
import argparse


def generate(args):
    WIDTH = args.width
    HEIGHT = args.height
    GAMMA_P1 = args.gamma_p1
    GAMMA_P2 = args.gamma_p2
    STEP = args.step
    PHSSTEP = int(WIDTH/8)
    OUTDIR = args.output_dir

    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    imgs = []

    print('Generating sinusoidal patterns ...')
    angle_vel = 2*np.pi/PHSSTEP
    gamma = [1/GAMMA_P1, 1/GAMMA_P2]
    xs = np.array(range(WIDTH))
    for i in range(1, 3):
        for phs in range(1, 4):
            vec = 0.5*(np.cos(xs*angle_vel + np.pi*(phs-2)*2/3)+1)
            vec = 255*(vec**gamma[i-1])
            vec = np.round(vec)
            img = np.zeros((HEIGHT, WIDTH), np.uint8)
            for y in range(HEIGHT):
                img[y, :] = vec
            imgs.append(img)

    ys = np.array(range(HEIGHT))
    for i in range(1, 3):
        for phs in range(1, 4):
            vec = 0.5*(np.cos(ys*angle_vel + np.pi*(phs-2)*2/3)+1)
            vec = 255*(vec**gamma[i-1])
            img = np.zeros((HEIGHT, WIDTH), np.uint8)
            for x in range(WIDTH):
                img[:, x] = vec
            imgs.append(img)

    print('Generating graycode patterns ...')
    gc_height = int((HEIGHT-1)/STEP)+1
    gc_width = int((WIDTH-1)/STEP)+1

    graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
    patterns = graycode.generate()[1]
    for pat in patterns:
        if STEP == 1:
            img = pat
        else:
            img = np.zeros((HEIGHT, WIDTH), np.uint8)
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    img[y, x] = pat[int(y/STEP), int(x/STEP)]
        imgs.append(img)
    imgs.append(255*np.ones((HEIGHT, WIDTH), np.uint8))  # white
    imgs.append(np.zeros((HEIGHT, WIDTH), np.uint8))     # black

    for i, img in enumerate(imgs):
        cv2.imwrite(OUTDIR+'/pat'+str(i).zfill(2)+'.png', img)

    print('Saving config file ...')
    fs = cv2.FileStorage(OUTDIR+'/config.xml', cv2.FILE_STORAGE_WRITE)
    fs.write('disp_width', WIDTH)
    fs.write('disp_height', HEIGHT)
    fs.write('gamma_p1', GAMMA_P1)
    fs.write('gamma_p2', GAMMA_P2)
    fs.write('step', STEP)
    fs.release()

    print('Done')


def decode(args):
    BLACKTHR = args.black_thr
    WHITETHR = args.white_thr
    INPUTPRE = args.input_prefix
    OUTDIR = args.output_dir

    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    fs = cv2.FileStorage(args.config_file, cv2.FILE_STORAGE_READ)
    DISP_WIDTH = int(fs.getNode('disp_width').real())
    DISP_HEIGHT = int(fs.getNode('disp_height').real())
    GAMMA_P1 = fs.getNode('gamma_p1').real()
    GAMMA_P2 = fs.getNode('gamma_p2').real()
    STEP = int(fs.getNode('step').real())
    PHSSTEP = int(DISP_WIDTH/8)
    fs.release()

    gc_width = int((DISP_WIDTH-1)/STEP)+1
    gc_height = int((DISP_HEIGHT-1)/STEP)+1
    graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
    graycode.setBlackThreshold(BLACKTHR)
    graycode.setWhiteThreshold(WHITETHR)

    print('Loading images ...')
    re_num = re.compile(r'(\d+)')

    def numerical_sort(text):
        return int(re_num.split(text)[-2])

    filenames = sorted(
        glob.glob(INPUTPRE+'*.png'), key=numerical_sort)
    if len(filenames) != graycode.getNumberOfPatternImages() + 14:
        print('Number of images is not right (right number is ' +
              str(graycode.getNumberOfPatternImages() + 14) + ')')
        return

    imgs = []
    for f in filenames:
        imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
    ps_imgs = imgs[0:12]
    gc_imgs = imgs[12:]
    black = gc_imgs.pop()
    white = gc_imgs.pop()
    CAM_WIDTH = white.shape[1]
    CAM_HEIGHT = white.shape[0]

    print('Decoding graycode ...')
    gc_map = np.zeros((CAM_HEIGHT, CAM_WIDTH, 2), np.int16)
    mask = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.uint8)
    target_map_x = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.float32)
    target_map_y = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.float32)
    angle_vel = 2*np.pi/PHSSTEP
    for y in range(CAM_HEIGHT):
        for x in range(CAM_WIDTH):
            if int(white[y, x]) - int(black[y, x]) <= BLACKTHR:
                continue
            err, proj_pix = graycode.getProjPixel(gc_imgs, x, y)
            if not err:
                pos = STEP*np.array(proj_pix)
                gc_map[y, x, :] = pos
                target_map_x[y, x] = angle_vel*pos[0]
                target_map_y[y, x] = angle_vel*pos[1]
                mask[y, x] = 255

    def decode_ps(pimgs, gamma=1.0):
        pimg1 = (pimgs[0].astype(np.float32)/255)**gamma
        pimg2 = (pimgs[1].astype(np.float32)/255)**gamma
        pimg3 = (pimgs[2].astype(np.float32)/255)**gamma
        return np.arctan2(
            np.sqrt(3)*(pimg1-pimg3), 2*pimg2-pimg1-pimg3)

    def res_func(xs, tx, ty, imgsx, imgsy):
        dx = decode_ps(imgsx, xs[0])
        dy = decode_ps(imgsy, xs[0])
        dif = (dx-tx+np.pi) % (2*np.pi) - np.pi
        dif += (dy-ty+np.pi) % (2*np.pi) - np.pi
        dif = dif**2
        return np.sum(dif)

    def dif_func(xs, tx, ty, imgsx, imgsy):
        eps = 1e-5
        return (res_func(xs+np.array([eps]), tx, ty, imgsx, imgsy) - res_func(xs, tx, ty, imgsx, imgsy))/eps

    print('Estimating gamma1-dash ...')
    x0 = 1.0
    sdir = 1.0 if dif_func([x0], target_map_x, target_map_y,
                           ps_imgs[0:3], ps_imgs[6:9]) < 0 else -1.0
    res = line_search(res_func, dif_func, np.array([x0]), np.array([sdir]), args=(
        target_map_x, target_map_y, ps_imgs[0:3], ps_imgs[6:9]))
    gamma1d = x0 + sdir*res[0]
    print(' ', gamma1d, ', residual :', res[3])

    print('Estimating gamma2-dash ...')
    x0 = 1.0
    sdir = 1.0 if dif_func([x0], target_map_x, target_map_y,
                           ps_imgs[3:6], ps_imgs[9:12]) < 0 else -1.0
    res = line_search(res_func, dif_func, np.array([x0]), np.array([sdir]), args=(
        target_map_x, target_map_y, ps_imgs[3:6], ps_imgs[9:12]))
    gamma2d = x0 + sdir*res[0]
    print(' ', gamma2d, ', residual :', res[3])

    gamma_a = (GAMMA_P1 - GAMMA_P2)/(gamma1d - gamma2d)
    gamma_b = (GAMMA_P1*gamma2d - gamma1d*GAMMA_P2)/(GAMMA_P1 - GAMMA_P2)
    gamma_p = (1 - gamma_b)*gamma_a
    print('  gamma a :', gamma_a)
    print('  gamma b :', gamma_b)

    print('Result')
    print('  gamma p :', gamma_p)

    print('Done')


def main():
    parser = argparse.ArgumentParser(
        description='Gamma correction of the pro-cam system\n',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers()

    parser_gen = subparsers.add_parser(
        'gen', help='generate patterns as images')
    parser_gen.add_argument('width', type=int, help='display width [pix]')
    parser_gen.add_argument('height', type=int,
                            help='display height [pix]')
    parser_gen.add_argument(
        'gamma_p1', type=float, help='gamma value 1 for correction (arbitrary value)')
    parser_gen.add_argument(
        'gamma_p2', type=float, help='gamma value 2 for correction (arbitrary value)')
    parser_gen.add_argument(
        '-step', type=int, default=1, help='block size of graycode [pix]')
    parser_gen.add_argument('output_dir', help='path to output files')
    parser_gen.set_defaults(func=generate)

    parser_dec = subparsers.add_parser(
        'dec', help='decode captured patterns')
    parser_dec.add_argument(
        'input_prefix', help='prefix of path to captured images')
    parser_dec.add_argument('config_file', help='path to config.xml')
    parser_dec.add_argument('output_dir', help='path to output files')
    parser_dec.add_argument('-black_thr', type=int, default=5, help='')
    parser_dec.add_argument('-white_thr', type=int, default=40, help='')
    parser_dec.set_defaults(func=decode)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
