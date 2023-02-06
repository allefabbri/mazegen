import numpy
from stl import mesh

class axis3d():
  x = [0.5, 0, 0]
  y = [0, 0.5, 0]
  z = [0, 0, 0.5]

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

def make_mesh(p1,p2,p3,p4):
  data = numpy.zeros(2, dtype=mesh.Mesh.dtype)
  data['vectors'][0] = numpy.array([
    p1,
    p2,
    p3,
  ])
  data['vectors'][1] = numpy.array([
    p1,
    p3,
    p4,
  ])
  wall = mesh.Mesh(data.copy())
  return wall
