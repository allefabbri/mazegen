import numpy
from stl import mesh

def make_wall(x1,y1,x2,y2,h):
  data = numpy.zeros(2, dtype=mesh.Mesh.dtype)
  data['vectors'][0] = numpy.array([
    [x1, y1, 0],
    [x2, y2, 0],
    [x2, y2, h],
  ])
  data['vectors'][1] = numpy.array([
    [x2, y2, h],
    [x1, y1, h],
    [x1, y1, 0],
  ])
  wall = mesh.Mesh(data.copy())
  return wall


def export_wall_stl(walls, outfile='maze_walls.stl', scale=1, height=0.5):
  meshdata = []
  for x1,y1,x2,y2 in walls:
    walli = make_wall(scale*x1,scale*y1,scale*x2,scale*y2,height)
    meshdata.append(walli.data.copy())

  mwall = mesh.Mesh(numpy.concatenate(meshdata))
  mwall.save(outfile)
