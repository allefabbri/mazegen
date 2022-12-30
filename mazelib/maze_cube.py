import numpy as np
import logging

from mazelib import AbstractMaze

# RECT MAZE
class CubeMaze(AbstractMaze):

    def __init__(self, w=3, shapetype='', **kwargs):
        super().__init__(**kwargs)
        self.log = logging.getLogger(self.__class__.__name__)
        self.w = w
        self.h = w

        self.shape = self.make_shape(shapetype, w, w)

        points = np.full((3*w,4*w,2), -1)
        yshift = 2*w
        for i in range(w):
            for j in range(w):
                for k in range(2*w+1):
                    points[i+k, j+yshift,:] = [ i+k, j+yshift ]
                for k in range(3*w+1):
                    points[i+w,j+k,:] = [ i+w, j+k ]

        xmin, xmax = points[:,:,0].min(), points[:,:,0].max()
        ymin, ymax = points[:,:,1].min(), points[:,:,1].max()
        self.bbox = [xmin,xmax,ymin,ymax]

        grid = set([ (i,j) for i in range(3*w) for j in range(4*w) if (points[i,j,:] != [-1,-1]).all() ])
        adlist = { (i,j) : set([ pos for pos in [
            (i-1,j),
            (i+1,j),
            (i,j+1),
            (i,j-1),
        ] if pos in grid ]) for i in range(3*w) for j in range(4*w) if (points[i,j,:] != [-1,-1]).all() }

        # faces naming
        #        back
        #          |
        # left - down  - right
        #          |
        #        front
        #          |
        #         top

        # wrap faces creating link in one direction only
        for i in range(w):
            adlist[(i,3*w-1)].add((w,4*w-1-i))       # left-back
            adlist[(i,2*w)].add((w,w+i))             # left-front

            adlist[(2*w+i,3*w-1)].add((2*w-1,3*w+i)) # back-right
            adlist[(2*w+i,2*w)].add((2*w-1,2*w-1-i)) # front-right

            adlist[(0,2*w+i)].add((w,w-1-i))         # left-top
            adlist[(3*w-1,2*w+i)].add((2*w-1,w-1-i)) # right-top

            adlist[(w+i,0)].add((w+i,4*w-1))         # top-back

        # then simmetrize
        for p1,v in adlist.items():
            for p2 in v:
                adlist[p2].add(p1)

        grid -= self.shape
        madlist = { p1:set([ p2 for p2 in link if p2 in grid ]) if p1 in grid else set() for p1, link in adlist.items() }

        self.l = 1
        self.mazetype = 'cube'
        self.points = points
        self.grid = grid
        self.adlist = adlist
        self.madlist = madlist
        self.start = (w,0)
        self.end = (2*w-1,3*w-1)#(w-1,w-1)

    def compute_wrap(self, p1, p2):
        p1, p2 = sorted([p1,p2])
        i1,j1 = p1
        i2,j2 = p2

        facesgrid = {
            (0,2): 'L',
            (1,0): 'T',
            (1,1): 'F',
            (1,2): 'D',
            (1,3): 'B',
            (2,2): 'R',
        }
        g1 = i1 // self.w, j1 // self.w
        f1 = facesgrid[g1]
        g2 = i2 // self.w, j2 // self.w
        f2 = facesgrid[g2]

        return f1 + f2

    # draw utils
    def decorate(self, ax):

        xmin,xmax,ymin,ymax = self.bbox
        d = self.l
        # ax.set_xlim([xmin - d, xmax + d])
        # ax.set_ylim([ymin - d, ymax + d])

        ax = self.smart_draw_region(self.start, ax, color='blue')
        ax = self.smart_draw_region(self.end, ax, color='red')

        return ax

    def smart_draw_link(self, p1, p2, ax, color='blue', marker=None, zorder=99, linewidth=1):
        p1, p2 = sorted([p1,p2])
        i1,j1 = p1
        i2,j2 = p2
        (x1,y1),(x2,y2) = self.get_coords(p1,p2)
        wrap = self.compute_wrap(p1,p2)
        if wrap == 'LT':
            k = (j1 - j2 - self.w - 1) // 2
            xt1, yt1 = x1 - 0.5, y1
            xt2, yt2 = x2 - 0.5, y2
            xt, yt = -0.5, 2*self.w-1-k
            xc, yc = -0.5, 2*self.w - 0.5
            r = k + 0.5
            pos = [ ( xc + r * np.cos(t), yc + r * np.sin(t)) for t in np.linspace(np.pi/2, 3*np.pi/2, 40) ]
            ax.plot((xt1,x1), (yt1,y1), color=color, linewidth=linewidth)
            ax.plot((xt2,x2), (yt2,y2), color=color, linewidth=linewidth)
            ax.plot((xt,xt2), (yt,yt2), color=color, linewidth=linewidth)
            ax.plot(*list(zip(*pos)), color=color, linewidth=linewidth)
        elif wrap == 'LB':
            xt1, yt1 = x1, y1 + 0.5
            xt2, yt2 = x2 - 0.5, y2
            ax.plot((x1,xt1), (y1,yt1), color=color, linewidth=linewidth)
            ax.plot((xt1,xt2), (yt1,yt2), color=color, linewidth=linewidth)
            ax.plot((xt2,x2), (yt2,y2), color=color, linewidth=linewidth)
        elif wrap == 'LF':
            xt1, yt1 = x1, y1 - 0.5
            xt2, yt2 = x2 - 0.5, y2
            ax.plot((x1,xt1), (y1,yt1), color=color, linewidth=linewidth)
            ax.plot((xt1,xt2), (yt1,yt2), color=color, linewidth=linewidth)
            ax.plot((xt2,x2), (yt2,y2), color=color, linewidth=linewidth)
        elif wrap == 'TB':
            k = i1 - self.w
            xt1, yt1 = i1, j1 - k - 1
            xt2, yt2 = i1, j2 + k + 1
            xc, yc = (xt1+xt2)/2, (yt1+yt2)/2
            r = (yt2-yt1)/2
            pos = [ ( xc + (2*self.w+1 + 2*k) * np.cos(t), yc + r * np.sin(t)) for t in np.linspace(np.pi/2, 3*np.pi/2, 40) ]
            ax.plot((xt1,x1), (yt1,y1), color=color, linewidth=linewidth)
            ax.plot((xt2,x2), (yt2,y2), color=color, linewidth=linewidth)
            ax.plot(*list(zip(*pos)), color=color, linewidth=linewidth)
        elif wrap == 'TR':
            k = p1[1]
            xt1, yt1 = x1 + 0.5, y1
            xt2, yt2 = x2 + 0.5, y2
            xt, yt = 3*self.w-1 + 0.5, self.w+k
            xc, yc = 3*self.w-1 + 0.5, 2*self.w - 0.5
            r = self.w - k - 0.5
            pos = [ ( xc + r * np.cos(t), yc + r * np.sin(t)) for t in np.linspace(-np.pi/2, np.pi/2, 40) ]
            ax.plot((xt1,x1), (yt1,y1), color=color, linewidth=linewidth)
            ax.plot((xt2,x2), (yt2,y2), color=color, linewidth=linewidth)
            ax.plot((xt,xt1), (yt,yt1), color=color, linewidth=linewidth)
            ax.plot(*list(zip(*pos)), color=color, linewidth=linewidth)
        elif wrap == 'FR':
            xt1, yt1 = x1 + 0.5, y1
            xt2, yt2 = x2, y2 - 0.5
            ax.plot((x1,xt1), (y1,yt1), color=color, linewidth=linewidth)
            ax.plot((xt1,xt2), (yt1,yt2), color=color, linewidth=linewidth)
            ax.plot((xt2,x2), (yt2,y2), color=color, linewidth=linewidth)
        elif wrap == 'BR':
            xt1, yt1 = x1 + 0.5, y1
            xt2, yt2 = x2, y2 + 0.5
            ax.plot((x1,xt1), (y1,yt1), color=color, linewidth=linewidth)
            ax.plot((xt1,xt2), (yt1,yt2), color=color, linewidth=linewidth)
            ax.plot((xt2,x2), (yt2,y2), color=color, linewidth=linewidth)
        elif wrap in ['LD', 'TF', 'FD', 'DB', 'DR'] or wrap[0] == wrap[1]:
            ax.plot((x1, x2), (y1, y2), marker=marker, color=color, zorder=zorder, linewidth=linewidth)
        else:
            raise Exception(f'Unknown link wrap {wrap}')

        return ax

    def smart_draw_wall(self, p1, p2, ax):
        (x1,y1),(x2,y2) = self.get_coords(p1,p2)
        wrap = self.compute_wrap(p1,p2)
        if wrap == 'LT':
            ax.plot((x1-0.5,x1-0.5),(y1-0.5,y1+0.5), marker=None, color='black')
            ax.plot((x2-0.5,x2-0.5),(y2-0.5,y2+0.5), marker=None, color='black')
        elif wrap == 'LB':
            ax.plot((x1-0.5,x1+0.5),(y1+0.5,y1+0.5), marker=None, color='black')
            ax.plot((x2-0.5,x2-0.5),(y2-0.5,y2+0.5), marker=None, color='black')
        elif wrap == 'LF':
            ax.plot((x1-0.5,x1+0.5),(y1-0.5,y1-0.5), marker=None, color='black')
            ax.plot((x2-0.5,x2-0.5),(y2-0.5,y2+0.5), marker=None, color='black')
        elif wrap == 'TB':
            ax.plot((x1-0.5,x1+0.5),(y1-0.5,y1-0.5), marker=None, color='black')
            ax.plot((x2-0.5,x2+0.5),(y2+0.5,y2+0.5), marker=None, color='black')
        elif wrap == 'TR':
            ax.plot((x1+0.5,x1+0.5),(y1-0.5,y1+0.5), marker=None, color='black')
            ax.plot((x2+0.5,x2+0.5),(y2-0.5,y2+0.5), marker=None, color='black')
        elif wrap == 'FR':
            ax.plot((x1+0.5,x1+0.5),(y1-0.5,y1+0.5), marker=None, color='black')
            ax.plot((x2-0.5,x2+0.5),(y2-0.5,y2-0.5), marker=None, color='black')
        elif wrap == 'BR':
            ax.plot((x1+0.5,x1+0.5),(y1-0.5,y1+0.5), marker=None, color='black')
            ax.plot((x2-0.5,x2+0.5),(y2+0.5,y2+0.5), marker=None, color='black')
        elif wrap in ['LD', 'TF', 'FD', 'DB', 'DR'] or wrap[0] == wrap[1]:
            zc = (x1+x2)/2 + (y1+y2)/2 * 1j
            v = y1-y2 + (x2-x1)*1j
            v /= abs(v)
            w1 = zc + v * 1 / 2
            w2 = zc - v * 1 / 2
            ax.plot((w1.real, w2.real), (w1.imag, w2.imag), marker=None, color='black')
        else:
            raise Exception(f'Unknown wall wrap {wrap}')

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

    def __drawmaze(self, show, solve, debug):
        print('debug')
        self.drawmaze_debug(show, solve, debug)