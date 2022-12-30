import argparse
import logging
import logging.config
import coloredlogs
import os, sys, json

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

def process_jobs(jobs):
    for j in jobs:
        maze = MazeFactory().get_maze(ttype, **j)
        maze.create(minlen='auto')
        maze.drawmaze(show=False, solve=False, debug=True)
        maze.drawmaze(show=False, solve=False, debug=False)
        maze.drawmaze(show=False, solve=True, debug=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--type', default='rect', choices=['rect', 'polar', 'voronoi', 'hex', '3d', 'cube'], help='Maze type selector')
    parser.add_argument('-W', '--w', default=5, type=int, help='Maze width in cells')
    parser.add_argument('-H', '--h', default=5, type=int, help='Maze height in cells')
    parser.add_argument('-D', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-S', '--show', action='store_true', help='Enable matplotlib graphical widget')
    parser.add_argument('-M', '--minlen', default=0, type=int, help='Set maze route minimum length')
    parser.add_argument('-O', '--outdir', default='mazes', help='Output directory')

    parser.add_argument('-s', '--solve', action='store_true', help='Enable solution generation, in combo with --show')
    parser.add_argument('-sh', '--shapetype', default='', choices=['', 'L', 'O', 'H'], help='Select rectangular type obstacle shapes')
    parser.add_argument('-pb', '--border-width', default=1, type=float, help='Set points random border width')
    parser.add_argument('-si', '--show-id', action='store_true', help='Enable node id label, in combo with --debug')
    parser.add_argument('-se', '--seed', default=-1, type=int, help='Set numpy random seed')
    parser.add_argument('-dr', '--dry-run', action='store_true', help='Enable dry run: generate maze and disable png creation')
    parser.add_argument('-m', '--mode', default='debug', help='Mode selector: debug|test|path/to/json')

    args = parser.parse_args()
    init_log()

    params = {
        'w': args.w,
        'h': args.h,
        'border_width': args.border_width,
        'outdir':args.outdir,
        'shapetype':args.shapetype,
        'show_id':args.show_id,
        'seed':args.seed,
    }

    if (m:=args.mode) == 'debug':
        maze = MazeFactory().get_maze(args.type, **params)
        maze.create(args.minlen)

        if not args.dry_run:
            if args.show:
                maze.drawmaze(show=args.show, solve=args.solve, debug=args.debug)
            else:
                maze.drawmaze(show=False, solve=False, debug=True)
                maze.drawmaze(show=False, solve=False, debug=False)
                maze.drawmaze(show=False, solve=True, debug=False)
    elif m == 'gif':
        maze = MazeFactory().get_maze(args.type, **params)
        maze.create(args.minlen)
        maze.make_gif()
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

            process_jobs(jobs)
    else:
        try:
            data = json.load(open(m))
            for ttype, tkwargs in data:
                jobs = [ tkwargs ] * 5
                process_jobs(jobs)
        except Exception as e:
            print(f'Error in processing json file : {e}')
