import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np
from scipy.spatial import Voronoi
import logging

from mazelib import AbstractMaze

## VORONOI MAZE
class VoronoiMaze(AbstractMaze):

    def __init__(self, w=5, h=5, border_width=1, shapetype='', **kwargs):
        super().__init__(**kwargs)
        self.log = logging.getLogger(self.__class__.__name__)
        self.w = w
        self.h = h

        self.shape = self.make_shape(shapetype, w, h)

        border = 2
        vw = w + 2 * border
        vh = h + 2 * border
        points = np.random.rand(vw,vh,2) * (1-border_width) + border_width / 2

        # map voronoi point index to grid 2-tuples
        vmap = {}
        vpoints = []
        step = 0.5
        for i in range(vw):
            for j in range(vh):
                u,v = points[i][j]
                #u,v = 0,0
                ox = step / 2 * ( 1 - 2 * (j % 2) )
                oy = step / 2 * ( 1 - 2 * (i % 2) )
                points[i,j,:] = [ i, j ]
                points[i,j,:] = [ i + ox + u * step, j ]
                points[i,j,:] = [ i + ox + u * step, j + oy + v * step ]
                vmap[(i,j)] = len(vpoints)
                vpoints.append([ c for c in points[i,j,:] ])
        vmapi = { v:k for k,v in vmap.items() }

        # compute voronoi
        vor = Voronoi(vpoints)
        self.vor = vor

        xmin, xmax = points[:,:,0].min(), points[:,:,0].max()
        ymin, ymax = points[:,:,1].min(), points[:,:,1].max()
        self.bbox = [xmin,xmax,ymin,ymax]

        grid = set([ (i,j) for i in range(vw) for j in range(vh) ])
        adlist = { (i,j) : [ pos for pos in [
            (i-1,j),
            (i+1,j),
            (i,j+1),
            (i,j-1),
        ] if pos in grid ] for i in range(vw) for j in range(vh) }

        # cut all external border connections
        mazegrid = set([ (i,j) for i in range(border,vw-border) for j in range(border,vh-border) ])
        mazegrid -= self.shape
        madlist = { p1:set([ p2 for p2 in link if p2 in mazegrid ]) if p1 in mazegrid else set() for p1, link in adlist.items() }

        # compute voronoi adlist
        vadlist = { (i,j):set() for i in range(vw) for j in range(vh)}
        for p1, p2 in self.vor.ridge_points:
            if p1 != -1 and p2 != -1:
                vadlist[vmapi[p1]].add(vmapi[p2])
                vadlist[vmapi[p2]].add(vmapi[p1])

        # container for voronoi links indexed by 2-tuples
        vlinks = { (i,j): {(i,j):-1 for i in range(vw) for j in range(vh)} for i in range(vw) for j in range(vh) }
        for lid, (p1, p2) in enumerate(self.vor.ridge_points):
            vlinks[vmapi[p1]][vmapi[p2]] = lid
            vlinks[vmapi[p2]][vmapi[p1]] = lid
        vlinksi = { lid: (z1,z2) for z1, row in vlinks.items() for z2, lid in row.items() }

        # container for voronoi walls indexed by 2-tuples
        vwalls = { (i,j): {(i,j):[] for i in range(vw) for j in range(vh)} for i in range(vw) for j in range(vh) }
        for lid, (p1, p2) in enumerate(self.vor.ridge_vertices):
            if p1 != -1 and p2 != -1:
                x1,y1 = self.vor.vertices[p1]
                x2,y2 = self.vor.vertices[p2]
                coords = [x1,y1,x2,y2]
            else:
                coords = []

            (i1,j1),(i2,j2) = vlinksi[lid]
            vwalls[(i1,j1)][(i2,j2)] = coords
            vwalls[(i2,j2)][(i1,j1)] = coords

        self.border = border
        self.mazetype = 'voronoi'
        self.points = points
        self.vpoints = vpoints
        self.adlist = vadlist
        self.madlist = madlist
        self.vwalls = vwalls
        self.vmap = vmap
        self.start = (border,border)
        self.end = (vw-border-1,vh-border-1)

    # draw utils
    def decorate(self, ax):

        xmin,xmax,ymin,ymax = self.bbox
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        ax = self.smart_draw_region(self.start, ax, color='blue')
        ax = self.smart_draw_region(self.end, ax, color='red')

        # add missing walls
        for i1,j1 in self.adlist:
            for i2,j2 in self.adlist[(i1,j1)]:
                if not len(self.madlist[(i1,j1)]):
                    if len(self.madlist[(i2,j2)]):
                        ax = self.smart_draw_wall((i1,j1), (i2,j2), ax)

        return ax

    def smart_draw_wall(self, p1, p2, ax):
        i1,j1 = p1
        i2,j2 = p2
        if len(self.vwalls[(i1,j1)][(i2,j2)]):
            x1,y1,x2,y2 = self.vwalls[(i1,j1)][(i2,j2)]
            ax.plot((x1,x2), (y1,y2), marker=None, color='black')

            # for stl export
            self.walls_data.append([x1,y1,x2,y2])

        return ax

    def smart_draw_region(self, p1, ax, color='green'):
        pi1 = self.vmap[p1]
        reg = self.vor.point_region[pi1]
        reg = self.vor.regions[reg]
        if len(reg):
            if not -1 in reg:
                plt.fill(*zip(*[ self.vor.vertices[i] for i in reg]), color=color, edgecolor=None, zorder=500, alpha=0.2)
        return ax
