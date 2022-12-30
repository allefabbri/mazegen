# mazegen

A simple python library to generate printable labyrinth.

## Algorithm Visualization
![maze algorithm](docs/algorithm.gif "Maze Algorithm")

## Rectangular Maze
![rectangular maze](docs/rect.png "Rectangular Maze")

## Polar Maze
![polar maze](docs/polar.png "Polar Maze")

## Hexagonal Maze
![hexagonal maze](docs/hexagonal.png "Hexagonal Maze")

## Voronoi Maze
![voronoi maze](docs/voronoi.png "Voronoi Maze")

## Cube Maze
![cube maze](docs/cube.png "Cube Maze")

## 3D Maze
![3d maze](docs/3d.png "3D Maze")

## Usage
```
usage: mazegen.py [-h] [-T {rect,polar,voronoi,hex,3d,cube}] [-W W] [-H H] [-D]
                  [-S] [-M MINLEN] [-O OUTDIR] [-s] [-sh {,L,O,H}]
                  [-vb VORONOI_BORDER] [-du] [-si] [-se SEED] [-dr]
                  [-m {debug,test,book}]

optional arguments:
  -h, --help            show this help message and exit
  -T {rect,polar,voronoi,hex,3d,cube}, --type {rect,polar,voronoi,hex,3d,cube}
  -W W, --w W
  -H H, --h H
  -D, --debug
  -S, --show
  -M MINLEN, --minlen MINLEN
  -O OUTDIR, --outdir OUTDIR
  -s, --solve
  -sh {,L,O,H}, --shapetype {,L,O,H}
  -vb VORONOI_BORDER, --voronoi-border VORONOI_BORDER
  -du, --disable-uniq
  -si, --show-id
  -se SEED, --seed SEED
  -dr, --dry-run
  -m {debug,test,book}, --mode {debug,test,book}
```