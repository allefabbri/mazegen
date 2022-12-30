import matplotlib.pyplot as plt
from collections import deque
import random
from datetime import datetime
import os
from glob import glob
import logging
import numpy as np

from mazelib import AbstractMaze

## 3D MAZE
class TridiMaze(AbstractMaze):

    def __init__(self, w, h, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = logging.getLogger(self.__class__.__name__)
        self.w = w
        self.h = h
        self.d = d = 3
        self.l = 1

        points = np.random.rand(w,h,d,3)
        for i in range(w):
            for j in range(h):
                for k in range(d):
                    points[i,j,k,:] = [ i, j, k ]

        xmin, xmax = points[:,:,:,0].min(), points[:,:,:,0].max()
        ymin, ymax = points[:,:,:,1].min(), points[:,:,:,1].max()
        self.bbox = [xmin,xmax,ymin,ymax]

        grid = set([ (i,j,k) for i in range(w) for j in range(h) for k in range(d) ])
        adlist = { (i,j,k) : [ pos for pos in [
            (i-1,j,k),
            (i+1,j,k),
            (i,j+1,k),
            (i,j-1,k),
            (i,j,k+1),
            (i,j,k-1),
        ] if pos in grid ] for i in range(w) for j in range(h) for k in range(d) }

        self.mazetype = '3d'
        self.points = points
        self.adlist = adlist
        self.madlist = adlist
        self.start = (0,0,0)
        self.end = (w-1,h-1,d-1)

    def get_coords(self, *args):
        coords = []
        for arg in args:
            i,j,k = arg
            x,y,z = self.points[i,j,k,:]
            coords.append([x,y,z])
        return coords

    def drawmaze(self, show, solve, debug):
        d = self.d

        # for p,v in self.adlist.items(): print(p, v)
        # for p,v in self.admaze.items(): print(p, v)

        fig = plt.figure(figsize=self.figsize)
        axis = [ fig.add_subplot(1,d,z+1) for z in range(d) ]

        for p1 in self.adlist:
            # if self.show_id and debug:
            #     x1,y1,z1 = self.points[i1,j1,k1,:]
            #     ax.text(x1, y1 + 0.2, f'{(i1,j1,k1)}', bbox=dict(facecolor='white', edgecolor='black'), ha='center', zorder=300)

            for p2 in self.adlist[p1]:
                if debug:
                    axis = self.smart_draw_link(p1, p2, axis, marker='o', color='blue')

                if p2 not in self.admaze[p1]:
                    if len(self.admaze[p1]):
                        axis = self.smart_draw_wall(p1, p2, axis)

        # add stairs
        for p1, p2 in self.path:
            if p1[2] != p2[2]:
                axis = self.smart_draw_link(p1, p2, axis, color='black', marker=None)

            # if (i2,j2) not in self.admaze[(i1,j1)]:
            #     if len(self.admaze[(i1,j1)]):
            #         ax = self.smart_draw_wall((i1,j1), (i2,j2), ax)

        if debug:

            for p1 in self.madlist:
                for p2 in self.madlist[p1]:
                    axis = self.smart_draw_link(p1, p2, axis, marker='o', color='orange')

            for p1, p2 in self.path:
                axis = self.smart_draw_link(p1, p2, axis, color='red', zorder=100)

        if solve:

            stack = set()
            for p1,p2 in self.route:
                ax = self.smart_draw_link(p1, p2, axis, color='green', zorder=104)
                for p in [p1,p2]:
                    if p not in stack:
                        axis = self.smart_draw_region(p, axis, color='green')
                        stack.add(p)


        #ax = self.decorate(ax, w, h, route[0][-1], route[-1][0])
        axis = self.decorate(axis, debug)



        if show:
            plt.show()
        else:
            outfile = self.compose_filename(solve, debug)
            self.log.info(f'Saved : {outfile}')
            plt.savefig(outfile, bbox_inches='tight', dpi=1000)
        plt.close()

    # draw utils
    def decorate(self, axis, debug):

        xmin,xmax,ymin,ymax = self.bbox
        d = self.l

        for i, ax in enumerate(axis):

            if not debug:
                ax.axis(False)
                ax.set_title(f'Floor {i+1}')

            ax.set_aspect(1)

            ax.set_xlim([xmin - d, xmax + d])
            ax.set_ylim([ymin - d, ymax + d])


            # add missing walls
            delta = d / 2
            ax.plot(( xmin - delta, xmax + delta), ( ymin - delta, ymin - delta), marker=None, color='black')
            ax.plot(( xmin - delta, xmax + delta), ( ymax + delta, ymax + delta), marker=None, color='black')
            ax.plot(( xmin - delta, xmin - delta), ( ymin - delta, ymax + delta), marker=None, color='black')
            ax.plot(( xmax + delta, xmax + delta), ( ymin - delta, ymax + delta), marker=None, color='black')

        axis = self.smart_draw_region(self.start, axis, color='blue')
        axis = self.smart_draw_region(self.end, axis, color='red')

        return axis

    def smart_draw_link(self, p1, p2, ax, color='blue', marker=None, zorder=99):
        i1,j1,k1 = p1
        i2,j2,k2 = p2
        x1,y1,z1 = self.points[i1,j1,k1,:]
        x2,y2,z2 = self.points[i2,j2,k2,:]
        if k1 == k2:
            ax[k1].plot((x1, x2), (y1, y2), marker=marker, color=color, zorder=zorder)
        else:
            pl, ph = sorted([p1,p2], key=lambda p:p[2])
            (xl, yl, _), (xh, yh, _) = self.get_coords(pl,ph)
            delta = 0.15
            ax[pl[2]].plot((xl-delta, xl+delta), (yl-delta, yl+delta), marker=None, color=color, zorder=zorder)
            ax[ph[2]].plot((xh-delta, xh+delta), (yh+delta, yh-delta), marker=None, color=color, zorder=zorder)

        return ax

    def smart_draw_wall(self, p1, p2, ax):
        i1,j1,k1 = p1
        i2,j2,k2 = p2
        if k1 == k2:
            x1,y1,z1 = self.points[i1,j1,k1,:]
            x2,y2,z2 = self.points[i2,j2,k2,:]
            #print(x1,y1,x2,y2)
            zc = (x1+x2)/2 + (y1+y2)/2 * 1j
            v = y1-y2 + (x2-x1)*1j
            v /= abs(v)
            w1 = zc + v * 1 / 2
            w2 = zc - v * 1 / 2
            ax[k1].plot((w1.real, w2.real), (w1.imag, w2.imag), marker=None, color='black')
        return ax

    def smart_draw_region(self, p1, axis, color='blue', alpha=0.2):
        i1,j1,k1=p1
        x1,y1,_=self.points[i1,j1,k1,:]
        delta = 0.5
        polygon = [
            [ x1-delta, y1-delta ],
            [ x1-delta, y1+delta ],
            [ x1+delta, y1+delta ],
            [ x1+delta, y1-delta ],
        ]
        axis[k1].fill(*list(map(list, zip(*polygon))), color=color, alpha=alpha, edgecolor=None)
        return axis

