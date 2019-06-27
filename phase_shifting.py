# Copyright (c) 2019 kamino410. All rights reserved.
# This code is licensed under MIT license (see LICENSE.txt for details)

import sys
import os
import os.path
import re
import glob

import cv2
import numpy as np
import argparse
import plotly.offline as po
import plotly.graph_objs as go


def generate(args):
    WIDTH = args.width
    HEIGHT = args.height
    STEP = args.step
    GC_STEP = int(STEP/2)
    OUTDIR = args.output_dir

    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    imgs = []

    print('Generating sinusoidal patterns ...')
    angle_vel = np.array((6, 4))*np.pi/STEP
    xs = np.array(range(WIDTH))
    for i in range(1, 3):
        for phs in range(1, 4):
            vec = 128 + 127*np.cos(xs*angle_vel[i-1] + np.pi*(phs-2)*2/3)
            img = np.zeros((HEIGHT, WIDTH), np.uint8)
            for y in range(HEIGHT):
                img[y, :] = vec
            imgs.append(img)

    ys = np.array(range(HEIGHT))
    for i in range(1, 3):
        for phs in range(1, 4):
            vec = 128 + 127*np.cos(ys*angle_vel[i-1] + np.pi*(phs-2)*2/3)
            img = np.zeros((HEIGHT, WIDTH), np.uint8)
            for x in range(WIDTH):
                img[:, x] = vec
            imgs.append(img)

    print('Generating graycode patterns ...')
    gc_height = int((HEIGHT-1)/GC_STEP)+1
    gc_width = int((WIDTH-1)/GC_STEP)+1

    graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
    patterns = graycode.generate()[1]
    for pat in patterns:
        img = np.zeros((HEIGHT, WIDTH), np.uint8)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                img[y, x] = pat[int(y/GC_STEP), int(x/GC_STEP)]
        imgs.append(img)
    imgs.append(255*np.ones((HEIGHT, WIDTH), np.uint8))  # white
    imgs.append(np.zeros((HEIGHT, WIDTH), np.uint8))     # black

    for i, img in enumerate(imgs):
        cv2.imwrite(OUTDIR+'/pat'+str(i).zfill(2)+'.png', img)

    print('Saving config file ...')
    fs = cv2.FileStorage(OUTDIR+'/config.xml', cv2.FILE_STORAGE_WRITE)
    fs.write('disp_width', WIDTH)
    fs.write('disp_height', HEIGHT)
    fs.write('step', STEP)
    fs.release()

    print('Done')


def decode(args):
    BLACKTHR = args.black_thr
    WHITETHR = args.white_thr
    INPUTPRE = args.input_prefix
    FILTER = args.filter_size
    OUTPUTDIR = args.output_dir

    fs = cv2.FileStorage(args.config_file, cv2.FILE_STORAGE_READ)
    DISP_WIDTH = int(fs.getNode('disp_width').real())
    DISP_HEIGHT = int(fs.getNode('disp_height').real())
    STEP = int(fs.getNode('step').real())
    GC_STEP = int(STEP/2)
    fs.release()

    gc_width = int((DISP_WIDTH-1)/GC_STEP)+1
    gc_height = int((DISP_HEIGHT-1)/GC_STEP)+1
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

    def decode_ps(pimgs):
        pimg1 = pimgs[0].astype(np.float64)
        pimg2 = pimgs[1].astype(np.float64)
        pimg3 = pimgs[2].astype(np.float64)
        return np.arctan2(
            np.sqrt(3)*(pimg1-pimg3), 2*pimg2-pimg1-pimg3)

    ps_map_x1 = decode_ps(ps_imgs[0:3])
    ps_map_x2 = decode_ps(ps_imgs[3:6])
    ps_map_y1 = decode_ps(ps_imgs[6:9])
    ps_map_y2 = decode_ps(ps_imgs[9:12])

    gc_map = np.zeros((CAM_HEIGHT, CAM_WIDTH, 2), np.int16)
    mask = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.uint8)
    for y in range(CAM_HEIGHT):
        for x in range(CAM_WIDTH):
            if int(white[y, x]) - int(black[y, x]) <= BLACKTHR:
                continue
            err, proj_pix = graycode.getProjPixel(gc_imgs, x, y)
            if not err:
                gc_map[y, x, :] = np.array(proj_pix)
                mask[y, x] = 255

    if FILTER != 0:
        print('Applying smoothing filter ...')
        ext_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    np.ones((FILTER*2+1, FILTER*2+1)))
        for y in range(CAM_HEIGHT):
            for x in range(CAM_WIDTH):
                if mask[y, x] == 0 and ext_mask[y, x] != 0:
                    sum_x = 0
                    sum_y = 0
                    cnt = 0
                    for dy in range(-FILTER, FILTER+1):
                        for dx in range(-FILTER, FILTER+1):
                            ty = y + dy
                            tx = x + dx
                            if ((dy != 0 or dx != 0)
                                    and ty >= 0 and ty < CAM_HEIGHT
                                    and tx >= 0 and tx < CAM_WIDTH
                                    and mask[ty, tx] != 0):
                                sum_x += gc_map[ty, tx, 0]
                                sum_y += gc_map[ty, tx, 1]
                                cnt += 1
                    if cnt != 0:
                        gc_map[y, x, 0] = np.round(sum_x/cnt)
                        gc_map[y, x, 1] = np.round(sum_y/cnt)

        mask = ext_mask

    def decode_pixel(gc, ps1, ps2):
        dif = None
        if ps1 > ps2:
            if ps1-ps2 > np.pi*4/3:
                dif = (ps2-ps1)+2*np.pi
            else:
                dif = ps1-ps2
        else:
            if ps2-ps1 > np.pi*4/3:
                dif = (ps1-ps2)+2*np.pi
            else:
                dif = ps2-ps1

        p = None
        if gc % 2 == 0:
            p = ps1
            if dif > np.pi/6 and p < 0:
                p = p + 2*np.pi
            if dif > np.pi/2 and p < np.pi*7/6:
                p = p + 2*np.pi
        else:
            p = ps1
            if dif > np.pi*5/6 and p > 0:
                p = p - 2*np.pi
            if dif < np.pi/2 and p < np.pi/6:
                p = p + 2*np.pi
            p = p + np.pi
        return gc*GC_STEP + STEP*p/3/2/np.pi

    print('Decoding each pixels ...')
    viz = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)
    res_x = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.float64)
    res_y = np.zeros((CAM_HEIGHT, CAM_WIDTH), np.float64)
    for y in range(CAM_HEIGHT):
        for x in range(CAM_WIDTH):
            if mask[y, x] != 0:
                est_x = decode_pixel(
                    gc_map[y, x, 0], ps_map_x1[y, x], ps_map_x2[y, x])
                est_y = decode_pixel(
                    gc_map[y, x, 1], ps_map_y1[y, x], ps_map_y2[y, x])

                viz[y, x, :] = (est_x, est_y, 128)
                res_x[y, x] = est_x
                res_y[y, x] = est_y

    cv2.imwrite(OUTPUTDIR+'/vizualized.png', viz)

    # data = []
    # xs = np.array(range(CAM_WIDTH))
    # data.append(go.Scatter(
    #     x=xs, y=res_x[150, :], mode='lines', name='res1'))
    # data.append(go.Scatter(x=xs, y=res[200, :, 0], mode='lines', name='res2'))

    # po.plot(data, filename='test.html')


def main():
    parser = argparse.ArgumentParser(
        description='3-step phase shifting method\n',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers()

    parser_gen = subparsers.add_parser(
        'gen', help='generate patterns as images')
    parser_gen.add_argument('width', type=int, help='display width [pix]')
    parser_gen.add_argument('height', type=int,
                            help='display height [pix]')
    parser_gen.add_argument(
        'step', type=int, help='block size of graycode [pix]')
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
    parser_dec.add_argument('-filter_size', type=int, default=0,
                            help='half size of smoothing filter for graycode '
                            '(0->None, 1->3x3, 2->5x5)')
    parser_dec.set_defaults(func=decode)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
