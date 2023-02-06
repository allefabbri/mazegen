import numpy as np
import logging

from mazelib import AbstractMaze

# HEX MAZE
class HexMaze(AbstractMaze):

    def __init__(self, w=5, h=10, shapetype='', **kwargs):
        super().__init__(**kwargs)
        self.log = logging.getLogger(self.__class__.__name__)
        self.w = w
        self.h = h

        self.shape = self.make_shape(shapetype, w, h)

        border = 2
        w = w + 2 * border
        h = h + 2 * border
        points = np.zeros((w,h,2))
        l = 1
        dx = 3 * l / 2
        dy = np.sqrt(3) / 2 * l
        for i in range(w):
            for j in range(h):
                points[i,j,:] = [ dx * i, j * dy ]

        xmin, xmax = points[:,:,0].min(), points[:,:,0].max()
        ymin, ymax = points[:,:,1].min(), points[:,:,1].max()
        self.bbox = [xmin,xmax,ymin,ymax]

        grid = set([ (i,j) for i in range((j%2),w,2) for j in range(0,h,2) ])
        grid |= set([ (i,j) for i in range(1-(j%2),w,2) for j in range(1,h,2) ])
        adlist = { (i,j) : set([ pos for pos in [
            (i-1,j-1),
            (i-1,j+1),
            (i+1,j-1),
            (i+1,j+1),
            (i,j-2),
            (i,j+2),
        ] if pos in grid ]) for i in range(w) for j in range(h) }
        # for p,v in adlist.items(): print(p,v)

        mazegrid = set([ (i,j) for i in range(border,w-border) for j in range(border,h-border) ])
        mazegrid -= self.shape
        madlist = { p1:set([ p2 for p2 in link if p2 in mazegrid ]) if p1 in mazegrid else set() for p1, link in adlist.items() }

        self.l = l
        self.dx = dx
        self.dy = dy
        self.border = border
        self.mazetype = 'hex'
        self.points = points
        self.grid = grid
        self.adlist = adlist
        self.madlist = madlist
        self.start = min([ p for p,v in madlist.items() if len(v) ])
        self.end = max([ p for p,v in madlist.items() if len(v) ])

    # draw utils
    def decorate(self, ax):

        xmin,xmax,ymin,ymax = self.bbox
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

        ax = self.smart_draw_region(self.start, ax, color='blue')
        ax = self.smart_draw_region(self.end, ax, color='red')

        xs = [ x for x in np.arange(0, self.bbox[1] + 0.1, self.dx) ]
        ax.set_xticks(xs)
        ax.set_xticklabels([ f'x {x:.2f} \ni {int((x)/self.dx)}' for x in xs])

        ys = [ y for y in np.arange(0, self.bbox[3] + 0.1, self.dy) ]
        ax.set_yticks(ys)
        ax.set_yticklabels([ f'y {y:.2f} \ni {int((y)/self.dy)}' for y in ys])

        # add missing walls
        for p1 in self.adlist:
            for p2 in self.adlist[p1]:
                if not len(self.madlist[p1]):
                    if len(self.madlist[p2]):
                        ax = self.smart_draw_wall(p1, p2, ax)

        return ax

    def smart_draw_wall(self, p1, p2, ax):
        i1,j1 = p1
        i2,j2 = p2
        x1,y1 = self.points[i1,j1,:]
        x2,y2 = self.points[i2,j2,:]
        #print(x1,y1,x2,y2)
        zc = (x1+x2)/2 + (y1+y2)/2 * 1j
        v = y1-y2 + (x2-x1)*1j
        v /= abs(v)
        w1 = zc + v * self.l / 2
        w2 = zc - v * self.l / 2
        ax.plot((w1.real, w2.real), (w1.imag, w2.imag), marker=None, color='black')

        # for stl export
        self.walls_data.append([w1.real, w1.imag, w2.real, w2.imag])

        return ax

    def smart_draw_region(self, p1, ax, color='blue', alpha=0.2):
        i1,j1 = p1
        x1,y1 = self.points[i1,j1,:]
        dtheta = 2 * np.pi / 6
        polygon = [
            [ x1 + self.l * np.cos(theta), y1 + self.l * np.sin(theta) ] for theta in np.arange(0, 2*np.pi, dtheta)
        ]
        ax.fill(*list(map(list, zip(*polygon))), color=color, alpha=alpha, edgecolor=None)
        return ax
