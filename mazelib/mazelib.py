import matplotlib.pyplot as plt
from matplotlib import animation, rc
from collections import deque
import random
from datetime import datetime
import os
from glob import glob
import logging
import numpy as np

## MAZE BFS
def mazebfs(adlist, start, end):
    # tmetric = datetime.now()
    log = logging.getLogger('mazebfs')

    if start not in adlist:
        raise Exception(f'Start {start} not in adlist')
    else:
        if not len(adlist[start]):
            raise Exception(f'Start {start} not connected')

    if end not in adlist:
        raise Exception(f'End {end} not in adlist')
    else:
        if not len(adlist[end]):
            raise Exception(f'End {end} not in connected')

    # BFS iterative version
    # implemented with queue
    Q = deque()
    visited = set()
    path = []
    found = False
    Q.append(start)
    while Q:
        current = Q.popleft()
        visited.add(current)
        while True:
            available = [ c for c in adlist[current] if c not in visited ]
            if len(available):
                idx = int(np.random.rand() * len(available) )
                newpos = available[idx]
                path.append([ current, newpos ])
                visited.add(newpos)

                if newpos == end:
                    found = True
                    break

                Q.append(newpos)
                current = newpos
            else:
                break

    # collect route to end
    if found:
        try:
            routeidx = path.index([ i for i in path if i[1] == end ][0])
            route = [path[routeidx]]
            while path[routeidx][0] != start:
                routeidx = path.index([ i for i in path if i[1] == path[routeidx][0] ][0])
                route.append(path[routeidx])
        except Exception as e:
            log.error(f'Error in collect route : {e}')
            route = []
    else:
        raise Exception(f'Path {start} {end} not found')

    # compute maze adjacency list
    admaze = { p:set() for p in adlist }

    try:
        for p1,p2 in path:
            admaze[p1].add(p2)
            admaze[p2].add(p1)
    except Exception as e:
        raise Exception(f'Error in collect path : {e}')

    # tmetric = datetime.now() - tmetric
    # log.debug(f'bfs took {tmetric}')
    return admaze, path, route

## MAZE DRAWER
class MazeDrawer():

    def __init__(self, show_id=False, outdir='mazes', **kwargs):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.figsize = (10,10)
        self.show_id = show_id

    def create(self, minlen):
        tmetric = datetime.now()

        if isinstance(minlen, str):
            self.minlen = 3 * int(np.sqrt(self.w*self.h))
        else:
            self.minlen = minlen

        maxlen = 0
        created = 0
        while True:
            mazedata = mazebfs(self.madlist, start=self.start, end=self.end)
            created += 1
            if (c:=len(mazedata[2])) >= self.minlen:
                self.log.info(f'Maze {self.w}x{self.h} created (tries {created}, minlen {self.minlen}), route {len(mazedata[2])}')
                break
            elif created > 100:
                self.log.error(f'Route len {minlen} not found in {created} tries, maxlen {maxlen} ')
                exit(128)
            else:
                maxlen = maxlen if maxlen > c else c
                if not created % 50:
                    self.log.warning(f'Try {created} maxlen {maxlen} < {self.minlen}')


        self.admaze, self.path, self.route = mazedata

        tmetric = datetime.now() - tmetric
        self.log.debug(f'Maze creation took {tmetric}')

    def compose_filename(self, solve, debug):
        outfile = f'{self.outdir}/maze-{self.mazetype}-{self.w:03d}x{self.h:03d}'

        outfiles = glob(outfile + '*.png')

        suffix = ''
        if solve: suffix = '-solved' + suffix
        if debug: suffix = '-debug' + suffix
        if suffix == '': suffix = '-labyrinth'
        outfiles = [ f for f in outfiles if suffix in f ]

        mid = self.npseed
        outfile += f'-{mid:04d}{suffix}.png'

        return outfile

    def drawmaze(self, show, solve, debug):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1,1,1)

        # loop over base grid adlist
        stack = set()
        for p1 in self.adlist:
            [(x1,y1)] = self.get_coords(p1)

            if self.show_id and debug:
                ax.text(x1, y1 + 0.2, f'{p1}', bbox=dict(facecolor='white', edgecolor='black'), ha='center', zorder=300)

            for p2 in self.adlist[p1]:
                if (p1,p2) not in stack and (p2,p1) not in stack:

                    if debug:
                        ax = self.smart_draw_link(p1, p2, ax, marker='o', color='blue', zorder=0)

                    if p2 not in self.admaze[p1]:
                        if len(self.admaze[p1]):
                            ax = self.smart_draw_wall(p1, p2, ax)
                            stack.add((p1, p2))
                            stack.add((p2, p1))

        if debug:

            ax.set_title(f'Size {self.w}x{self.h} Route Len {len(self.route)} Seed {self.npseed}')

            for p1 in self.madlist:
                for p2 in self.madlist[p1]:
                    ax = self.smart_draw_link(p1, p2, ax, marker='o', color='orange', zorder=1)

            for stepn, (p1,p2) in enumerate(self.path):
                if [p1,p2] in self.route or [p2,p1] in self.route:
                    color = 'green'
                    zorder = 101
                else:
                    color = 'red'
                    zorder = 100
                ax = self.smart_draw_link(p1, p2, ax, color=color, zorder=zorder, linewidth=5)

                # display bfs step text
                #(x1,y1),(x2,y2) = self.get_coords(p1,p2)
                #ax.text((x1+x2)/2+0.1, (y1+y2)/2+0.1, f'{stepn}', bbox=dict(facecolor='white', edgecolor='black'), ha='center', zorder=300)

        if solve:

            stack = set()
            for p1,p2 in self.route:
                ax = self.smart_draw_link(p1, p2, ax, color='green', zorder=104)
                for p in [p1,p2]:
                    if p not in stack:
                        ax = self.smart_draw_region(p, ax, color='green')
                        stack.add(p)

        ax = self.decorate(ax)

        ax.set_aspect(1)

        if not debug:
            ax.axis(False)

        if show:
            plt.show()
        else:
            outfile = self.compose_filename(solve, debug)
            self.log.info(f'Saved : {outfile}')
            plt.savefig(outfile, bbox_inches='tight')
        plt.close()

    def drawmaze_debug(self, show, solve, debug):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1,1,1)

        # loop over base grid adlist
        link_cnt = 0
        link_cache = set()
        for p1 in self.adlist:
            [(x1,y1)] = self.get_coords(p1)
            ax.scatter(x1, y1, color='blue')
            for p2 in self.adlist[p1]:
                if (p1,p2) not in link_cache and (p2,p1) not in link_cache:
                    ax = self.smart_draw_link(p1, p2, ax, marker='o', color='blue', zorder=0)
                    link_cnt += 1
                    link_cache.add((p1,p2))
                    link_cache.add((p2,p1))

                    ax = self.smart_draw_wall(p1, p2, ax)


        self.log.debug(f'Links drawn : {link_cnt}')

        # import numpy as np
        # for i in np.arange(-0.5, w, 1): ax.plot((i,i), (-0.5,h-0.5), color='black')
        # for j in np.arange(-0.5, h, 1): ax.plot((-0.5,w-0.5), (j,j), color='black')
        # ax.set_aspect(1)

        # ax = self.decorate(ax)

        if not debug:
            ax.axis(False)

        if show:
            plt.show()
        else:
            outfile = self.compose_filename(solve, debug)
            self.log.info(f'Saved : {outfile}')
            plt.savefig(outfile, bbox_inches='tight')
        plt.close()

    def animate(self, i):
        self.ax.clear()
        self.ax.axis(False)
        self.ax = self.decorate(self.ax)

        wall_destroyed = set()
        for k, (p1,p2) in enumerate(self.path[:i]):
            self.ax = self.smart_draw_link(p1,p2,self.ax, color='red', linewidth=2)
            wall_destroyed.add(tuple(sorted([p1,p2])))

        for p1 in self.adlist:
            for p2 in self.adlist[p1]:
                if tuple(sorted([p1,p2])) not in wall_destroyed:
                    self.ax = self.smart_draw_wall(p1,p2, self.ax)

    def make_gif(self):
        rc('animation', html='html5')

        fig, ax = plt.subplots()

        max_step = len(self.path) + 1
        max_time = 10

        self.ax = ax
        anim = animation.FuncAnimation(
            fig,
            self.animate,
            frames=max_step,
            interval=1000*max_time/max_step,
        )

        outfile = self.compose_filename(False, False).replace('-labyrinth.png', '.gif')
        anim.save(f'{outfile}', writer=animation.ImageMagickWriter(fps=max_step/max_time, extra_args=['-loop', '1']))


## ABSTRACT MAZE
class AbstractMaze(MazeDrawer):

    def __init__(self, *args, seed=-1, **kwargs):
        super().__init__(**kwargs)
        self.log = logging.getLogger(self.__class__.__name__)

        self.npseed = self.set_seed(seed)
        self.log.debug(f'Numpy seed : {self.npseed}')

        self.mazetype = 'base'
        self.points = []
        self.adlist = {}
        self.start = (0,0)
        self.end = (0,0)

    def set_seed(self, seed):
        if seed != -1:
            npseed = seed
        else:
            npseed = int(random.random() * 9999)
        np.random.seed(npseed)
        return npseed

    def get_coords(self, *args):
        coords = []
        for arg in args:
            i,j = arg
            x,y = self.points[i,j,:]
            coords.append([x,y])
        return coords

    def make_shape(self, shapetype, w, h):
        if shapetype == '':
            shape = set()
        elif shapetype == 'L':
            shape = set((i,j) for i in range(w//2,w) for j in range(h//2))
        elif shapetype == 'O':
            shape = set((i,j) for i in range(w//3,(2*w)//3) for j in range(h//3,(2*h)//3))
        elif shapetype == 'H':
            shape = set((i,j) for i in range(w//4,w//4+2) for j in range((2*h)//8,(7*h)//8))
            shape |= set((i,j) for i in range((3*w)//4,(3*w)//4+2) for j in range((2*h)//8,(7*h)//8))
        else:
            raise Exception(f'Shape type {shapetype} not supported')
        return shape

    def decorate(self, ax):
        return ax

    def smart_draw_link(self, p1, p2, ax, color='blue', marker=None, zorder=99, linewidth=1):
        (x1,y1),(x2,y2) = self.get_coords(p1,p2)
        ax.plot((x1, x2), (y1, y2), marker=marker, color=color, zorder=zorder, linewidth=linewidth)
        return ax

    def smart_draw_wall(self, p1, p2, ax):
        (x1,y1),(x2,y2) = self.get_coords(p1,p2)
        ax.scatter((x1+x2)/2, (y1+y2)/2, marker='+', color='black')
        return ax

    def smart_draw_region(self, p1, ax, color='blue', alpha=0.2):
        (x1,y1) = self.get_coords(p1)
        ax.scatter(x1, y1, marker='x', color='black')
        return ax

# MAZE FACTORY
class MazeFactory():

    def get_maze(self, mazetype, **kwargs):
        if mazetype == 'rect':
            from maze_rect import RectMaze
            maze = RectMaze(**kwargs)
        elif mazetype == 'polar':
            from maze_polar import PolarMaze
            maze = PolarMaze(**kwargs)
        elif mazetype == 'voronoi':
            from maze_voronoi import VoronoiMaze
            maze = VoronoiMaze(**kwargs)
        elif mazetype == 'hex':
            from maze_hex import HexMaze
            maze = HexMaze(**kwargs)
        elif mazetype == '3d':
            from maze_3d import TridiMaze
            maze = TridiMaze(**kwargs)
        elif mazetype == 'cube':
            from maze_cube import CubeMaze
            maze = CubeMaze(**kwargs)
        return maze
