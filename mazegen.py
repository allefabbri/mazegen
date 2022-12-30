import argparse
import logging
import logging.config
import coloredlogs
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'mazelib'))
from mazelib import MazeFactory

def init_log():
    console_formatter = coloredlogs.ColoredFormatter('%(asctime)s [%(levelname)s] (%(name)s) %(message)s')
    console_handler = logging.StreamHandler() # Add Handler for console output (stderr) - use StreamHandler(sys.stdout) to use stdout instead
    console_handler.setFormatter(console_formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            console_handler # console stream handler
        ]
    )

    # silence specific logger
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--type', default='rect', choices=['rect', 'polar', 'voronoi', 'hex', '3d', 'cube'])
    parser.add_argument('-W', '--w', default=5, type=int)
    parser.add_argument('-H', '--h', default=5, type=int)
    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-S', '--show', action='store_true')
    parser.add_argument('-M', '--minlen', default=0, type=int)
    parser.add_argument('-O', '--outdir', default='mazes')

    parser.add_argument('-s', '--solve', action='store_true')
    parser.add_argument('-sh', '--shapetype', default='', choices=['', 'L', 'O', 'H'])
    parser.add_argument('-vb', '--voronoi-border', default=1, type=float)
    parser.add_argument('-du', '--disable-uniq', action='store_true')
    parser.add_argument('-si', '--show-id', action='store_true')
    parser.add_argument('-se', '--seed', default=-1, type=int)
    parser.add_argument('-dr', '--dry-run', action='store_true')
    parser.add_argument('-m', '--mode', default='debug', choices=['debug', 'test', 'book'])

    args = parser.parse_args()
    init_log()

    if (m:=args.mode) == 'debug':
        params = {
            'w': args.w,
            'h': args.h,
            'border_width': args.voronoi_border,
            'outdir':args.outdir,
            'disable_uniq':args.disable_uniq,
            'shapetype':args.shapetype,
            'show_id':args.show_id,
            'seed':args.seed,
        }
        maze = MazeFactory().get_maze(args.type, **params)
        maze.create(args.minlen)

        if not args.dry_run:
            if args.show:
                maze.drawmaze(show=args.show, solve=args.solve, debug=args.debug)
            else:
                maze.drawmaze(show=False, solve=False, debug=True)
                maze.drawmaze(show=False, solve=False, debug=False)
                maze.drawmaze(show=False, solve=True, debug=False)
    elif m == 'test':
        npseed = 0
        testsuite = [
            [
                'rect',
                {
                    'outdir': 'test',
                    'seed': npseed,
                    'w': 10,
                    'h': 10,
                    'border_width': 1,
                },
            ],
            [
                'polar',
                {
                    'outdir': 'test',
                    'seed': npseed,
                    'w': 10,
                    'h': 10,
                },
            ],
            [
                'hex',
                {
                    'outdir': 'test',
                    'seed': npseed,
                    'w': 10,
                    'h': 20,
                },
            ],
            [
                'voronoi',
                {
                    'outdir': 'test',
                    'seed': npseed,
                    'w': 10,
                    'h': 10,
                },
            ],
            [
                'cube',
                {
                    'outdir': 'test',
                    'seed': npseed,
                    'w': 10,
                },
            ],
            [
                '3d',
                {
                    'outdir': 'test',
                    'seed': npseed,
                    'w': 10,
                    'h': 10,
                },
            ]
        ]
        for ttype, tkwargs in testsuite:
            jobs = [ tkwargs ]
            if ttype == 'rect':
                for sh in [ 'L', 'H', 'O']:
                    tmp = { k:v for k,v in tkwargs.items() }
                    tmp.update({'shapetype':sh})
                    jobs.append(tmp)

            for j in jobs:
                maze = MazeFactory().get_maze(ttype, **j)
                maze.create(minlen=0)
                maze.drawmaze(show=False, solve=False, debug=True)
                maze.drawmaze(show=False, solve=False, debug=False)
                maze.drawmaze(show=False, solve=True, debug=False)
    elif m == 'book':
        outdir = 'xmas'
        data = [
            [
                'polar',
                {
                    'outdir': outdir,
                    'w': 10,
                    'h': 20,
                },
            ],
            [
                'rect',
                {
                    'outdir': outdir,
                    'w': 20,
                    'h': 20,
                },
            ],
            [
                'hex',
                {
                    'outdir': outdir,
                    'w': 40,
                    'h': 80,
                },
            ],
            [
                'voronoi',
                {
                    'outdir': outdir,
                    'w': 50,
                    'h': 50,
                },
            ]
        ]
        for ttype, tkwargs in data:
            jobs = [ tkwargs ] * 5
            for j in jobs:
                maze = MazeFactory().get_maze(ttype, **j)
                maze.create(minlen='auto')
                maze.drawmaze(show=False, solve=False, debug=True)
                maze.drawmaze(show=False, solve=False, debug=False)
                maze.drawmaze(show=False, solve=True, debug=False)