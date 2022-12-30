import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np
import logging

from mazelib import AbstractMaze

## POLAR MAZE
class PolarMaze(AbstractMaze):

    def __init__(self, *args, w=5, h=5, shapetype='', **kwargs):
        super().__init__(*args, **kwargs)
        self.log = logging.getLogger(self.__class__.__name__)
        self.w = w
        self.h = h

        self.shape = self.make_shape(shapetype, w, h)

        dtheta = 2 * np.pi / w
        points = np.zeros((w,h,4))
        for i in range(w):
            for j in range(h):
                points[i,j,:] = [ (1+j)*np.cos( (i+0.5)*dtheta ) , (1+j)*np.sin( (i+0.5)*dtheta ), 1+j, (i+0.5)*dtheta ]
        xmin, xmax = points[:,:,0].min(), points[:,:,0].max()
        ymin, ymax = points[:,:,1].min(), points[:,:,1].max()
        self.bbox = [xmin,xmax,ymin,ymax]

        grid = set([ (i,j) for i in range(w) for j in range(h) ])
        adlist = { (i,j) : [ pos for pos in [
            ( (i-1) % w ,j),
            ( (i+1) % w ,j),
            ( i,j+1),
            ( i,j-1),
        ] if pos in grid ] for i in range(w) for j in range(h) }

        grid -= self.shape
        madlist = { p1:set([ p2 for p2 in link if p2 in grid ]) if p1 in grid else set() for p1, link in adlist.items() }

        self.mazetype = 'polar'
        self.points = points
        self.adlist = adlist
        self.madlist = madlist
        self.start = (0,0)
        self.end = (w-1,h-1)

    def get_coords(self, *args):
        coords = []
        for arg in args:
            i,j = arg
            x,y,_,_ = self.points[i,j,:]
            coords.append([x,y])
        return coords

    # draw utils
    def decorate(self, ax):
        w = self.w
        h = self.h
        rmin = 0.5
        rmax = h + .5
        dtheta = 2 * np.pi / w
        ax.add_patch(Arc((0,0), 2*(rmin), 2*(rmin), angle=0, theta1=np.degrees(dtheta), theta2=360))
        ax.add_patch(Arc((0,0), 2*(rmax), 2*(rmax), angle=0, theta1=0, theta2=360))

        ax.set_xlim([-h-1,h+1])
        ax.set_ylim([-h-1,h+1])

        marker_radius = 0.35
        i,j = self.end
        x,y,_,_ = self.points[i,j,:]
        ax.add_artist(plt.Circle((x,y), marker_radius, facecolor='green', zorder=200))
        ax.add_artist(plt.Circle((0,0), marker_radius, facecolor='blue', zorder=200))
        i,j = self.start
        x,y,_,_ = self.points[i,j,:]
        ax.plot((0,x), (0,y), color='green', zorder=199)
        return ax

    def smart_draw_link(self, p1, p2, ax, color='blue', marker=None, zorder=99, linewidth=1):
        i1,j1 = p1
        i2,j2 = p2
        x1,y1,r1,theta1 = self.points[i1,j1,:]
        x2,y2,r2,theta2 = self.points[i2,j2,:]

        if i1 == i2:
            ax.plot((x1, x2), (y1, y2), marker=None, color=color, linewidth=linewidth)
        elif j1 == j2:
            if abs(i1-i2) > 1:
                cw = -np.sign(i2-i1)
            else:
                cw = np.sign(i2-i1)

            dt = abs(theta2-theta1)
            if dt > np.pi: dt = 2 * np.pi - dt
            pos = [ ( r1*np.cos(theta1+cw*t), r1*np.sin(theta1+cw*t)) for t in np.linspace(0, dt, 20) ]
            ax.plot(*list(zip(*pos)), color=color, linewidth=linewidth)
        else:
            raise Exception('nope')

        if marker is not None:
            ax.scatter((x1,x2),(y1,y2), s=100, c=color, zorder=zorder)

        return ax

    def smart_draw_wall(self, p1, p2, ax):
        i1,j1 = p1
        i2,j2 = p2
        _,_,r1,theta1 = self.points[i1,j1,:]
        _,_,r2,theta2 = self.points[i2,j2,:]

        if i1 == i2:
            dtheta = 2 * np.pi / self.w
            t1 = theta1 - dtheta / 2
            t2 = theta1 + dtheta / 2
            r = (r1+r2) / 2
            ax.add_patch(Arc((0,0), 2*(r), 2*(r), angle=0, theta1=np.degrees(t1), theta2=np.degrees(t2), edgecolor='black'))
        elif j1 == j2:
            if abs(i1-i2) > 1:
                newtheta = 0
            else:
                newtheta = (theta1+theta2)/2
            xw1 = (r1 - 0.5) * np.cos( newtheta )
            yw1 = (r1 - 0.5) * np.sin( newtheta )
            xw2 = (r1 + 0.5) * np.cos( newtheta )
            yw2 = (r1 + 0.5) * np.sin( newtheta )
            ax.plot((xw1, xw2), (yw1, yw2), marker=None, color='black')
            pass
        else:
            raise Exception('nope')

        return ax

    def smart_draw_region(self, p1, ax, color='blue', alpha=0.2):
        return ax
