import numpy as np
import logging

from mazelib import AbstractMaze

# RECT MAZE
class RectMaze(AbstractMaze):

    def __init__(self, w=5, h=5, border_width=1, shapetype='', **kwargs):
        super().__init__(**kwargs)
        self.log = logging.getLogger(self.__class__.__name__)
        self.w = w
        self.h = h

        border = 1
        w = w + 2 * border
        h = h + 2 * border

        self.shape = self.make_shape(shapetype, w, h)

        points = np.random.rand(w,h,2) * (1-border_width) + border_width / 2
        for i in range(w):
            for j in range(h):
                u,v = points[i][j]
                points[i,j,:] = [ i+u, j+v ]

        xmin, xmax = points[:,:,0].min(), points[:,:,0].max()
        ymin, ymax = points[:,:,1].min(), points[:,:,1].max()
        self.bbox = [xmin,xmax,ymin,ymax]

        grid = set([ (i,j) for i in range(w) for j in range(h) ])
        adlist = { (i,j) : [ pos for pos in [
            (i-1,j),
            (i+1,j),
            (i,j+1),
            (i,j-1),
        ] if pos in grid ] for i in range(w) for j in range(h) }

        mazegrid = set([ (i,j) for i in range(1,w-1) for j in range(1,h-1) ])
        mazegrid -= self.shape
        madlist = { p1:set([ p2 for p2 in link if p2 in mazegrid ]) if p1 in mazegrid else set() for p1, link in adlist.items() }

        self.l = 1
        self.mazetype = 'rect' + shapetype
        self.points = points
        self.grid = grid
        self.adlist = adlist
        self.madlist = madlist
        self.start = (1,1)
        self.end = (w-2,h-2)

    # draw utils
    def decorate(self, ax):

        xmin,xmax,ymin,ymax = self.bbox
        d = self.l
        ax.set_xlim([xmin - d, xmax + d])
        ax.set_ylim([ymin - d, ymax + d])

        ax = self.smart_draw_region(self.start, ax, color='blue')
        ax = self.smart_draw_region(self.end, ax, color='red')

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
        w1 = zc + v * 1 / 2
        w2 = zc - v * 1 / 2
        ax.plot((w1.real, w2.real), (w1.imag, w2.imag), marker=None, color='black')

        # for stl export
        self.walls_data.append([w1.real, w1.imag, w2.real, w2.imag])

        return ax

    def smart_draw_region(self, p1, ax, color='blue', alpha=0.2):
        i1,j1=p1
        x1,y1=self.points[i1,j1,:]
        delta = 0.5
        polygon = [
            [ x1-delta, y1-delta ],
            [ x1-delta, y1+delta ],
            [ x1+delta, y1+delta ],
            [ x1+delta, y1-delta ],
        ]
        ax.fill(*list(map(list, zip(*polygon))), color=color, alpha=alpha, edgecolor=None)
        return ax
