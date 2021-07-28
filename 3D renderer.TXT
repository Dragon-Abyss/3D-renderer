import json
import math
import numpy as np
from PIL import Image
texture = Image.open('/content/sample_data/texture.png')
xres, yres = texture.size
render1 = Image.new('RGB', [512,512], 0x000000)
render2 = Image.new('RGB', [512,512], 0x000000)
render3 = Image.new('RGB', [512,512], 0x000000)
render4 = Image.new('RGB', [512,512], 0x000000)
render5 = Image.new('RGB', [512,512], 0x000000)
render6 = Image.new('RGB', [512,512], 0x000000)
render = Image.new('RGB', [512,512], 0x000000)
width, height = render.size

for y in range(height):
  for x in range(width):
    render.putpixel( (x, y), (0, 0, 0) )

def getColorPhong(a, b, c):
  N0 = np.array([a[0][0], a[1][0], a[2][0]])
  N0 = N0/np.linalg.norm(N0)
  N1 = np.array([b[0][0], b[1][0], b[2][0]])
  N1 = N1/np.linalg.norm(N1)
  N2 = np.array([c[0][0], c[1][0], c[2][0]])
  N2 = N2/np.linalg.norm(N2)
  N = alpha * N0 + beta * N1 + gamma * N2

  v0 = np.array([x0, y0, z0])
  v1 = np.array([x1, y1, z1])
  v2 = np.array([x2, y2, z2])
  v = alpha * v0 + beta * v1 + gamma * v2
  E = -v/np.linalg.norm(v)

  R = 2 * (N.dot(L)) * N - L
  R = R/np.linalg.norm(R)

  c = 0.5 * (R.dot(E))**20 * le + 0.4 * N.dot(L) * le + 0.3 * la + 0.8 * textureLookup(t_pixel[0], t_pixel[1], texture)

  return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))

def textureLookup(u, v, texmap):
  xLocation = u * (xres - 1)
  yLocation = v * (yres - 1)

  p00 = (math.floor(xLocation), math.floor(yLocation))
  p11 = (math.ceil(xLocation), math.ceil(yLocation))
  p10 = (math.ceil(xLocation), math.floor(yLocation))
  p01 = (math.floor(xLocation), math.ceil(yLocation))

  f = xLocation - math.floor(xLocation)
  g = yLocation - math.floor(yLocation)
  p00RGB = np.array([texmap.getpixel(p00)[0], texmap.getpixel(p00)[1], texmap.getpixel(p00)[2]])
  p11RGB = np.array([texmap.getpixel(p11)[0], texmap.getpixel(p11)[1], texmap.getpixel(p11)[2]])
  p10RGB = np.array([texmap.getpixel(p10)[0], texmap.getpixel(p10)[1], texmap.getpixel(p10)[2]])
  p01RGB = np.array([texmap.getpixel(p01)[0], texmap.getpixel(p01)[1], texmap.getpixel(p01)[2]])

  p0010RGB = (p10RGB - p00RGB) * f + p00RGB
  p0111RGB = (p11RGB - p01RGB) * f + p11RGB
  RGB = (p0111RGB - p0010RGB) * g + p0010RGB

  return RGB/255

def getCamMat(f, t):
  n = np.array([0, 0, 0])
  for i in range(3):
    n[i] = f[i] - t[i]
  n = n/np.linalg.norm(n)
  v = np.array([0, 1, 0])
  u = np.cross(v, n)
  u = u/np.linalg.norm(u)
  v = np.cross(n, u)
  return np.array([[u[0], u[1], u[2], (f.dot(u))], [v[0], v[1], v[2], (f.dot(v))], [n[0], n[1], n[2], (f.dot(n))], [0, 0, 0, 1]])

def mxp(M, p):
  x = M[0][0]*p[0][0] + M[0][1]*p[1][0] + M[0][2]*p[2][0] + M[0][3]*p[3][0]
  y = M[1][0]*p[0][0] + M[1][1]*p[1][0] + M[1][2]*p[2][0] + M[1][3]*p[3][0]
  z = M[2][0]*p[0][0] + M[2][1]*p[1][0] + M[2][2]*p[2][0] + M[2][3]*p[3][0]
  w = M[3][0]*p[0][0] + M[3][1]*p[1][0] + M[3][2]*p[2][0] + M[3][3]*p[3][0]
  return np.array([[x],[y],[z],[w]])

def f01(a, b):
  return (y0-y1)*a + (x1-x0)*b + x0*y1-x1*y0

def f12(a, b):
  return (y1-y2)*a + (x2-x1)*b + x1*y2-x2*y1

def f20(a, b):
  return (y2-y0)*a + (x0-x2)*b + x2*y0-x0*y2

camFrom = np.array([8, -6, 15])
camTo = np.array([0, 0, 0])
VM = getCamMat(camFrom, camTo)
light = np.array([20, 10, 10])
L = light/np.linalg.norm(light)
le = np.array([1, 1, 1])
la = np.array([0.5, 0.5, 0.5])
near = 20
far = 50
left = -5
right = 5
top = 5
bottom = -5
P = np.array([[2*near/(right-left), 0, -1*(right+left)/(right-left), 0], [0, 2*near/(top-bottom), -1*(top+bottom)/(top-bottom), 0], [0, 0, (far+near)/(far-near), (far*near)/(far-near)], [0, 0, -1, 0]])

with open ('/content/sample_data/teapot.json') as triangles:
  triangle_dict = json.load(triangles)

#AA1
z_buffer = [[math.inf for i in range(512)] for j in range(512)]

for i in range(896):
  x0 = triangle_dict['data'][i]['v0']['v'][0]
  y0 = triangle_dict['data'][i]['v0']['v'][1]
  z0 = triangle_dict['data'][i]['v0']['v'][2]
  v0u = 1 - triangle_dict['data'][i]['v0']['t'][0]
  v0v = triangle_dict['data'][i]['v0']['t'][1]
  x1 = triangle_dict['data'][i]['v1']['v'][0]
  y1 = triangle_dict['data'][i]['v1']['v'][1]
  z1 = triangle_dict['data'][i]['v1']['v'][2]
  v1u = 1 - triangle_dict['data'][i]['v1']['t'][0]
  v1v = triangle_dict['data'][i]['v1']['t'][1]
  x2 = triangle_dict['data'][i]['v2']['v'][0]
  y2 = triangle_dict['data'][i]['v2']['v'][1]
  z2 = triangle_dict['data'][i]['v2']['v'][2]
  v2u = 1 - triangle_dict['data'][i]['v2']['t'][0]
  v2v = triangle_dict['data'][i]['v2']['t'][1]

  x0n = triangle_dict['data'][i]['v0']['n'][0]
  y0n = triangle_dict['data'][i]['v0']['n'][1]
  z0n = triangle_dict['data'][i]['v0']['n'][2]
  x1n = triangle_dict['data'][i]['v1']['n'][0]
  y1n = triangle_dict['data'][i]['v1']['n'][1]
  z1n = triangle_dict['data'][i]['v1']['n'][2]
  x2n = triangle_dict['data'][i]['v2']['n'][0]
  y2n = triangle_dict['data'][i]['v2']['n'][1]
  z2n = triangle_dict['data'][i]['v2']['n'][2]

  v0 = np.array([[x0], [y0], [z0], [1]])
  v1 = np.array([[x1], [y1], [z1], [1]])
  v2 = np.array([[x2], [y2], [z2], [1]])
  v0n = np.array([[x0n], [y0n], [z0n], [1]])
  v1n = np.array([[x1n], [y1n], [z1n], [1]])
  v2n = np.array([[x2n], [y2n], [z2n], [1]])

  v0 = mxp(VM, v0)
  v1 = mxp(VM, v1)
  v2 = mxp(VM, v2)

  v0 = mxp(P, v0)
  v1 = mxp(P, v1)
  v2 = mxp(P, v2)

  v0[0][0] = v0[0][0]/v0[3][0]
  v0[1][0] = v0[1][0]/v0[3][0]
  v0[2][0] = v0[2][0]/v0[3][0]
  v1[0][0] = v1[0][0]/v1[3][0]
  v1[1][0] = v1[1][0]/v1[3][0]
  v1[2][0] = v1[2][0]/v1[3][0]
  v2[0][0] = v2[0][0]/v2[3][0]
  v2[1][0] = v2[1][0]/v2[3][0]
  v2[2][0] = v2[2][0]/v2[3][0]

  x0 = (v0[0][0]+1)*511/2
  y0 = (v0[1][0]+1)*511/2
  z0 = v0[2][0]
  x1 = (v1[0][0]+1)*511/2
  y1 = (v1[1][0]+1)*511/2
  z1 = v1[2][0]
  x2 = (v2[0][0]+1)*511/2
  y2 = (v2[1][0]+1)*511/2
  z2 = v2[2][0]

  x0 += -0.52
  y0 += 0.38
  x1 += -0.52
  y1 += 0.38
  x2 += -0.52
  y2 += 0.38

  v0t = np.array([v0u/z0, v0v/z0])
  v1t = np.array([v1u/z1, v1v/z1])
  v2t = np.array([v2u/z2, v2v/z2])

  xmin = math.floor(min(x0, x1, x2))
  xmax = math.ceil(max(x0, x1, x2))
  ymin = math.floor(min(y0, y1, y2))
  ymax = math.ceil(max(y0, y1, y2))

  for y in range(ymin, ymax):
    for x in range(xmin, xmax):
      alpha = f12(x,y) / f12(x0,y0)
      beta =  f20(x,y) / f20(x1,y1)
      gamma = f01(x,y) / f01(x2,y2)

      z_pixel = 1/(alpha * (1/z0) + beta * (1/z1) + gamma * (1/z2))
      t_pixel = (v0t * alpha + v1t * beta + v2t * gamma) * z_pixel

      if (alpha >= 0) and (beta >= 0) and (gamma >= 0):
        if (x >= 0 and x <= 511 and y >= 0 and y <= 511):
          if (z_pixel < z_buffer[x][y]):
            render1.putpixel( (x, y), getColorPhong(v0n, v1n, v2n) )
            z_buffer[x][y] = z_pixel

#AA2
z_buffer = [[math.inf for i in range(512)] for j in range(512)]

for i in range(896):
  x0 = triangle_dict['data'][i]['v0']['v'][0]
  y0 = triangle_dict['data'][i]['v0']['v'][1]
  z0 = triangle_dict['data'][i]['v0']['v'][2]
  v0u = 1 - triangle_dict['data'][i]['v0']['t'][0]
  v0v = triangle_dict['data'][i]['v0']['t'][1]
  x1 = triangle_dict['data'][i]['v1']['v'][0]
  y1 = triangle_dict['data'][i]['v1']['v'][1]
  z1 = triangle_dict['data'][i]['v1']['v'][2]
  v1u = 1 - triangle_dict['data'][i]['v1']['t'][0]
  v1v = triangle_dict['data'][i]['v1']['t'][1]
  x2 = triangle_dict['data'][i]['v2']['v'][0]
  y2 = triangle_dict['data'][i]['v2']['v'][1]
  z2 = triangle_dict['data'][i]['v2']['v'][2]
  v2u = 1 - triangle_dict['data'][i]['v2']['t'][0]
  v2v = triangle_dict['data'][i]['v2']['t'][1]

  x0n = triangle_dict['data'][i]['v0']['n'][0]
  y0n = triangle_dict['data'][i]['v0']['n'][1]
  z0n = triangle_dict['data'][i]['v0']['n'][2]
  x1n = triangle_dict['data'][i]['v1']['n'][0]
  y1n = triangle_dict['data'][i]['v1']['n'][1]
  z1n = triangle_dict['data'][i]['v1']['n'][2]
  x2n = triangle_dict['data'][i]['v2']['n'][0]
  y2n = triangle_dict['data'][i]['v2']['n'][1]
  z2n = triangle_dict['data'][i]['v2']['n'][2]

  v0 = np.array([[x0], [y0], [z0], [1]])
  v1 = np.array([[x1], [y1], [z1], [1]])
  v2 = np.array([[x2], [y2], [z2], [1]])
  v0n = np.array([[x0n], [y0n], [z0n], [1]])
  v1n = np.array([[x1n], [y1n], [z1n], [1]])
  v2n = np.array([[x2n], [y2n], [z2n], [1]])

  v0 = mxp(VM, v0)
  v1 = mxp(VM, v1)
  v2 = mxp(VM, v2)

  v0 = mxp(P, v0)
  v1 = mxp(P, v1)
  v2 = mxp(P, v2)

  v0[0][0] = v0[0][0]/v0[3][0]
  v0[1][0] = v0[1][0]/v0[3][0]
  v0[2][0] = v0[2][0]/v0[3][0]
  v1[0][0] = v1[0][0]/v1[3][0]
  v1[1][0] = v1[1][0]/v1[3][0]
  v1[2][0] = v1[2][0]/v1[3][0]
  v2[0][0] = v2[0][0]/v2[3][0]
  v2[1][0] = v2[1][0]/v2[3][0]
  v2[2][0] = v2[2][0]/v2[3][0]

  x0 = (v0[0][0]+1)*511/2
  y0 = (v0[1][0]+1)*511/2
  z0 = v0[2][0]
  x1 = (v1[0][0]+1)*511/2
  y1 = (v1[1][0]+1)*511/2
  z1 = v1[2][0]
  x2 = (v2[0][0]+1)*511/2
  y2 = (v2[1][0]+1)*511/2
  z2 = v2[2][0]

  x0 += 0.41
  y0 += 0.56
  x1 += 0.41
  y1 += 0.56
  x2 += 0.41
  y2 += 0.56

  v0t = np.array([v0u/z0, v0v/z0])
  v1t = np.array([v1u/z1, v1v/z1])
  v2t = np.array([v2u/z2, v2v/z2])

  xmin = math.floor(min(x0, x1, x2))
  xmax = math.ceil(max(x0, x1, x2))
  ymin = math.floor(min(y0, y1, y2))
  ymax = math.ceil(max(y0, y1, y2))

  for y in range(ymin, ymax):
    for x in range(xmin, xmax):
      alpha = f12(x,y) / f12(x0,y0)
      beta =  f20(x,y) / f20(x1,y1)
      gamma = f01(x,y) / f01(x2,y2)

      z_pixel = 1/(alpha * (1/z0) + beta * (1/z1) + gamma * (1/z2))
      t_pixel = (v0t * alpha + v1t * beta + v2t * gamma) * z_pixel

      if (alpha >= 0) and (beta >= 0) and (gamma >= 0):
        if (x >= 0 and x <= 511 and y >= 0 and y <= 511):
          if (z_pixel < z_buffer[x][y]):
            render2.putpixel( (x, y), getColorPhong(v0n, v1n, v2n) )
            z_buffer[x][y] = z_pixel

#AA3
z_buffer = [[math.inf for i in range(512)] for j in range(512)]

for i in range(896):
  x0 = triangle_dict['data'][i]['v0']['v'][0]
  y0 = triangle_dict['data'][i]['v0']['v'][1]
  z0 = triangle_dict['data'][i]['v0']['v'][2]
  v0u = 1 - triangle_dict['data'][i]['v0']['t'][0]
  v0v = triangle_dict['data'][i]['v0']['t'][1]
  x1 = triangle_dict['data'][i]['v1']['v'][0]
  y1 = triangle_dict['data'][i]['v1']['v'][1]
  z1 = triangle_dict['data'][i]['v1']['v'][2]
  v1u = 1 - triangle_dict['data'][i]['v1']['t'][0]
  v1v = triangle_dict['data'][i]['v1']['t'][1]
  x2 = triangle_dict['data'][i]['v2']['v'][0]
  y2 = triangle_dict['data'][i]['v2']['v'][1]
  z2 = triangle_dict['data'][i]['v2']['v'][2]
  v2u = 1 - triangle_dict['data'][i]['v2']['t'][0]
  v2v = triangle_dict['data'][i]['v2']['t'][1]

  x0n = triangle_dict['data'][i]['v0']['n'][0]
  y0n = triangle_dict['data'][i]['v0']['n'][1]
  z0n = triangle_dict['data'][i]['v0']['n'][2]
  x1n = triangle_dict['data'][i]['v1']['n'][0]
  y1n = triangle_dict['data'][i]['v1']['n'][1]
  z1n = triangle_dict['data'][i]['v1']['n'][2]
  x2n = triangle_dict['data'][i]['v2']['n'][0]
  y2n = triangle_dict['data'][i]['v2']['n'][1]
  z2n = triangle_dict['data'][i]['v2']['n'][2]

  v0 = np.array([[x0], [y0], [z0], [1]])
  v1 = np.array([[x1], [y1], [z1], [1]])
  v2 = np.array([[x2], [y2], [z2], [1]])
  v0n = np.array([[x0n], [y0n], [z0n], [1]])
  v1n = np.array([[x1n], [y1n], [z1n], [1]])
  v2n = np.array([[x2n], [y2n], [z2n], [1]])

  v0 = mxp(VM, v0)
  v1 = mxp(VM, v1)
  v2 = mxp(VM, v2)

  v0 = mxp(P, v0)
  v1 = mxp(P, v1)
  v2 = mxp(P, v2)

  v0[0][0] = v0[0][0]/v0[3][0]
  v0[1][0] = v0[1][0]/v0[3][0]
  v0[2][0] = v0[2][0]/v0[3][0]
  v1[0][0] = v1[0][0]/v1[3][0]
  v1[1][0] = v1[1][0]/v1[3][0]
  v1[2][0] = v1[2][0]/v1[3][0]
  v2[0][0] = v2[0][0]/v2[3][0]
  v2[1][0] = v2[1][0]/v2[3][0]
  v2[2][0] = v2[2][0]/v2[3][0]

  x0 = (v0[0][0]+1)*511/2
  y0 = (v0[1][0]+1)*511/2
  z0 = v0[2][0]
  x1 = (v1[0][0]+1)*511/2
  y1 = (v1[1][0]+1)*511/2
  z1 = v1[2][0]
  x2 = (v2[0][0]+1)*511/2
  y2 = (v2[1][0]+1)*511/2
  z2 = v2[2][0]

  x0 += 0.27
  y0 += 0.08
  x1 += 0.27
  y1 += 0.08
  x2 += 0.27
  y2 += 0.08

  v0t = np.array([v0u/z0, v0v/z0])
  v1t = np.array([v1u/z1, v1v/z1])
  v2t = np.array([v2u/z2, v2v/z2])

  xmin = math.floor(min(x0, x1, x2))
  xmax = math.ceil(max(x0, x1, x2))
  ymin = math.floor(min(y0, y1, y2))
  ymax = math.ceil(max(y0, y1, y2))

  for y in range(ymin, ymax):
    for x in range(xmin, xmax):
      alpha = f12(x,y) / f12(x0,y0)
      beta =  f20(x,y) / f20(x1,y1)
      gamma = f01(x,y) / f01(x2,y2)

      z_pixel = 1/(alpha * (1/z0) + beta * (1/z1) + gamma * (1/z2))
      t_pixel = (v0t * alpha + v1t * beta + v2t * gamma) * z_pixel

      if (alpha >= 0) and (beta >= 0) and (gamma >= 0):
        if (x >= 0 and x <= 511 and y >= 0 and y <= 511):
          if (z_pixel < z_buffer[x][y]):
            render3.putpixel( (x, y), getColorPhong(v0n, v1n, v2n) )
            z_buffer[x][y] = z_pixel

#AA4
z_buffer = [[math.inf for i in range(512)] for j in range(512)]

for i in range(896):
  x0 = triangle_dict['data'][i]['v0']['v'][0]
  y0 = triangle_dict['data'][i]['v0']['v'][1]
  z0 = triangle_dict['data'][i]['v0']['v'][2]
  v0u = 1 - triangle_dict['data'][i]['v0']['t'][0]
  v0v = triangle_dict['data'][i]['v0']['t'][1]
  x1 = triangle_dict['data'][i]['v1']['v'][0]
  y1 = triangle_dict['data'][i]['v1']['v'][1]
  z1 = triangle_dict['data'][i]['v1']['v'][2]
  v1u = 1 - triangle_dict['data'][i]['v1']['t'][0]
  v1v = triangle_dict['data'][i]['v1']['t'][1]
  x2 = triangle_dict['data'][i]['v2']['v'][0]
  y2 = triangle_dict['data'][i]['v2']['v'][1]
  z2 = triangle_dict['data'][i]['v2']['v'][2]
  v2u = 1 - triangle_dict['data'][i]['v2']['t'][0]
  v2v = triangle_dict['data'][i]['v2']['t'][1]

  x0n = triangle_dict['data'][i]['v0']['n'][0]
  y0n = triangle_dict['data'][i]['v0']['n'][1]
  z0n = triangle_dict['data'][i]['v0']['n'][2]
  x1n = triangle_dict['data'][i]['v1']['n'][0]
  y1n = triangle_dict['data'][i]['v1']['n'][1]
  z1n = triangle_dict['data'][i]['v1']['n'][2]
  x2n = triangle_dict['data'][i]['v2']['n'][0]
  y2n = triangle_dict['data'][i]['v2']['n'][1]
  z2n = triangle_dict['data'][i]['v2']['n'][2]

  v0 = np.array([[x0], [y0], [z0], [1]])
  v1 = np.array([[x1], [y1], [z1], [1]])
  v2 = np.array([[x2], [y2], [z2], [1]])
  v0n = np.array([[x0n], [y0n], [z0n], [1]])
  v1n = np.array([[x1n], [y1n], [z1n], [1]])
  v2n = np.array([[x2n], [y2n], [z2n], [1]])

  v0 = mxp(VM, v0)
  v1 = mxp(VM, v1)
  v2 = mxp(VM, v2)

  v0 = mxp(P, v0)
  v1 = mxp(P, v1)
  v2 = mxp(P, v2)

  v0[0][0] = v0[0][0]/v0[3][0]
  v0[1][0] = v0[1][0]/v0[3][0]
  v0[2][0] = v0[2][0]/v0[3][0]
  v1[0][0] = v1[0][0]/v1[3][0]
  v1[1][0] = v1[1][0]/v1[3][0]
  v1[2][0] = v1[2][0]/v1[3][0]
  v2[0][0] = v2[0][0]/v2[3][0]
  v2[1][0] = v2[1][0]/v2[3][0]
  v2[2][0] = v2[2][0]/v2[3][0]

  x0 = (v0[0][0]+1)*511/2
  y0 = (v0[1][0]+1)*511/2
  z0 = v0[2][0]
  x1 = (v1[0][0]+1)*511/2
  y1 = (v1[1][0]+1)*511/2
  z1 = v1[2][0]
  x2 = (v2[0][0]+1)*511/2
  y2 = (v2[1][0]+1)*511/2
  z2 = v2[2][0]

  x0 += -0.17
  y0 += -0.29
  x1 += -0.17
  y1 += -0.29
  x2 += -0.17
  y2 += -0.29

  v0t = np.array([v0u/z0, v0v/z0])
  v1t = np.array([v1u/z1, v1v/z1])
  v2t = np.array([v2u/z2, v2v/z2])

  xmin = math.floor(min(x0, x1, x2))
  xmax = math.ceil(max(x0, x1, x2))
  ymin = math.floor(min(y0, y1, y2))
  ymax = math.ceil(max(y0, y1, y2))

  for y in range(ymin, ymax):
    for x in range(xmin, xmax):
      alpha = f12(x,y) / f12(x0,y0)
      beta =  f20(x,y) / f20(x1,y1)
      gamma = f01(x,y) / f01(x2,y2)

      z_pixel = 1/(alpha * (1/z0) + beta * (1/z1) + gamma * (1/z2))
      t_pixel = (v0t * alpha + v1t * beta + v2t * gamma) * z_pixel

      if (alpha >= 0) and (beta >= 0) and (gamma >= 0):
        if (x >= 0 and x <= 511 and y >= 0 and y <= 511):
          if (z_pixel < z_buffer[x][y]):
            render4.putpixel( (x, y), getColorPhong(v0n, v1n, v2n) )
            z_buffer[x][y] = z_pixel

#AA5
z_buffer = [[math.inf for i in range(512)] for j in range(512)]

for i in range(896):
  x0 = triangle_dict['data'][i]['v0']['v'][0]
  y0 = triangle_dict['data'][i]['v0']['v'][1]
  z0 = triangle_dict['data'][i]['v0']['v'][2]
  v0u = 1 - triangle_dict['data'][i]['v0']['t'][0]
  v0v = triangle_dict['data'][i]['v0']['t'][1]
  x1 = triangle_dict['data'][i]['v1']['v'][0]
  y1 = triangle_dict['data'][i]['v1']['v'][1]
  z1 = triangle_dict['data'][i]['v1']['v'][2]
  v1u = 1 - triangle_dict['data'][i]['v1']['t'][0]
  v1v = triangle_dict['data'][i]['v1']['t'][1]
  x2 = triangle_dict['data'][i]['v2']['v'][0]
  y2 = triangle_dict['data'][i]['v2']['v'][1]
  z2 = triangle_dict['data'][i]['v2']['v'][2]
  v2u = 1 - triangle_dict['data'][i]['v2']['t'][0]
  v2v = triangle_dict['data'][i]['v2']['t'][1]

  x0n = triangle_dict['data'][i]['v0']['n'][0]
  y0n = triangle_dict['data'][i]['v0']['n'][1]
  z0n = triangle_dict['data'][i]['v0']['n'][2]
  x1n = triangle_dict['data'][i]['v1']['n'][0]
  y1n = triangle_dict['data'][i]['v1']['n'][1]
  z1n = triangle_dict['data'][i]['v1']['n'][2]
  x2n = triangle_dict['data'][i]['v2']['n'][0]
  y2n = triangle_dict['data'][i]['v2']['n'][1]
  z2n = triangle_dict['data'][i]['v2']['n'][2]

  v0 = np.array([[x0], [y0], [z0], [1]])
  v1 = np.array([[x1], [y1], [z1], [1]])
  v2 = np.array([[x2], [y2], [z2], [1]])
  v0n = np.array([[x0n], [y0n], [z0n], [1]])
  v1n = np.array([[x1n], [y1n], [z1n], [1]])
  v2n = np.array([[x2n], [y2n], [z2n], [1]])

  v0 = mxp(VM, v0)
  v1 = mxp(VM, v1)
  v2 = mxp(VM, v2)

  v0 = mxp(P, v0)
  v1 = mxp(P, v1)
  v2 = mxp(P, v2)

  v0[0][0] = v0[0][0]/v0[3][0]
  v0[1][0] = v0[1][0]/v0[3][0]
  v0[2][0] = v0[2][0]/v0[3][0]
  v1[0][0] = v1[0][0]/v1[3][0]
  v1[1][0] = v1[1][0]/v1[3][0]
  v1[2][0] = v1[2][0]/v1[3][0]
  v2[0][0] = v2[0][0]/v2[3][0]
  v2[1][0] = v2[1][0]/v2[3][0]
  v2[2][0] = v2[2][0]/v2[3][0]

  x0 = (v0[0][0]+1)*511/2
  y0 = (v0[1][0]+1)*511/2
  z0 = v0[2][0]
  x1 = (v1[0][0]+1)*511/2
  y1 = (v1[1][0]+1)*511/2
  z1 = v1[2][0]
  x2 = (v2[0][0]+1)*511/2
  y2 = (v2[1][0]+1)*511/2
  z2 = v2[2][0]

  x0 += 0.58
  y0 += -0.55
  x1 += 0.58
  y1 += -0.55
  x2 += 0.58
  y2 += -0.55

  v0t = np.array([v0u/z0, v0v/z0])
  v1t = np.array([v1u/z1, v1v/z1])
  v2t = np.array([v2u/z2, v2v/z2])

  xmin = math.floor(min(x0, x1, x2))
  xmax = math.ceil(max(x0, x1, x2))
  ymin = math.floor(min(y0, y1, y2))
  ymax = math.ceil(max(y0, y1, y2))

  for y in range(ymin, ymax):
    for x in range(xmin, xmax):
      alpha = f12(x,y) / f12(x0,y0)
      beta =  f20(x,y) / f20(x1,y1)
      gamma = f01(x,y) / f01(x2,y2)

      z_pixel = 1/(alpha * (1/z0) + beta * (1/z1) + gamma * (1/z2))
      t_pixel = (v0t * alpha + v1t * beta + v2t * gamma) * z_pixel

      if (alpha >= 0) and (beta >= 0) and (gamma >= 0):
        if (x >= 0 and x <= 511 and y >= 0 and y <= 511):
          if (z_pixel < z_buffer[x][y]):
            render5.putpixel( (x, y), getColorPhong(v0n, v1n, v2n) )
            z_buffer[x][y] = z_pixel

#AA6
z_buffer = [[math.inf for i in range(512)] for j in range(512)]

for i in range(896):
  x0 = triangle_dict['data'][i]['v0']['v'][0]
  y0 = triangle_dict['data'][i]['v0']['v'][1]
  z0 = triangle_dict['data'][i]['v0']['v'][2]
  v0u = 1 - triangle_dict['data'][i]['v0']['t'][0]
  v0v = triangle_dict['data'][i]['v0']['t'][1]
  x1 = triangle_dict['data'][i]['v1']['v'][0]
  y1 = triangle_dict['data'][i]['v1']['v'][1]
  z1 = triangle_dict['data'][i]['v1']['v'][2]
  v1u = 1 - triangle_dict['data'][i]['v1']['t'][0]
  v1v = triangle_dict['data'][i]['v1']['t'][1]
  x2 = triangle_dict['data'][i]['v2']['v'][0]
  y2 = triangle_dict['data'][i]['v2']['v'][1]
  z2 = triangle_dict['data'][i]['v2']['v'][2]
  v2u = 1 - triangle_dict['data'][i]['v2']['t'][0]
  v2v = triangle_dict['data'][i]['v2']['t'][1]

  x0n = triangle_dict['data'][i]['v0']['n'][0]
  y0n = triangle_dict['data'][i]['v0']['n'][1]
  z0n = triangle_dict['data'][i]['v0']['n'][2]
  x1n = triangle_dict['data'][i]['v1']['n'][0]
  y1n = triangle_dict['data'][i]['v1']['n'][1]
  z1n = triangle_dict['data'][i]['v1']['n'][2]
  x2n = triangle_dict['data'][i]['v2']['n'][0]
  y2n = triangle_dict['data'][i]['v2']['n'][1]
  z2n = triangle_dict['data'][i]['v2']['n'][2]

  v0 = np.array([[x0], [y0], [z0], [1]])
  v1 = np.array([[x1], [y1], [z1], [1]])
  v2 = np.array([[x2], [y2], [z2], [1]])
  v0n = np.array([[x0n], [y0n], [z0n], [1]])
  v1n = np.array([[x1n], [y1n], [z1n], [1]])
  v2n = np.array([[x2n], [y2n], [z2n], [1]])

  v0 = mxp(VM, v0)
  v1 = mxp(VM, v1)
  v2 = mxp(VM, v2)

  v0 = mxp(P, v0)
  v1 = mxp(P, v1)
  v2 = mxp(P, v2)

  v0[0][0] = v0[0][0]/v0[3][0]
  v0[1][0] = v0[1][0]/v0[3][0]
  v0[2][0] = v0[2][0]/v0[3][0]
  v1[0][0] = v1[0][0]/v1[3][0]
  v1[1][0] = v1[1][0]/v1[3][0]
  v1[2][0] = v1[2][0]/v1[3][0]
  v2[0][0] = v2[0][0]/v2[3][0]
  v2[1][0] = v2[1][0]/v2[3][0]
  v2[2][0] = v2[2][0]/v2[3][0]

  x0 = (v0[0][0]+1)*511/2
  y0 = (v0[1][0]+1)*511/2
  z0 = v0[2][0]
  x1 = (v1[0][0]+1)*511/2
  y1 = (v1[1][0]+1)*511/2
  z1 = v1[2][0]
  x2 = (v2[0][0]+1)*511/2
  y2 = (v2[1][0]+1)*511/2
  z2 = v2[2][0]

  x0 += -0.31
  y0 += -0.71
  x1 += -0.31
  y1 += -0.71
  x2 += -0.31
  y2 += -0.71

  v0t = np.array([v0u/z0, v0v/z0])
  v1t = np.array([v1u/z1, v1v/z1])
  v2t = np.array([v2u/z2, v2v/z2])

  xmin = math.floor(min(x0, x1, x2))
  xmax = math.ceil(max(x0, x1, x2))
  ymin = math.floor(min(y0, y1, y2))
  ymax = math.ceil(max(y0, y1, y2))

  for y in range(ymin, ymax):
    for x in range(xmin, xmax):
      alpha = f12(x,y) / f12(x0,y0)
      beta =  f20(x,y) / f20(x1,y1)
      gamma = f01(x,y) / f01(x2,y2)

      z_pixel = 1/(alpha * (1/z0) + beta * (1/z1) + gamma * (1/z2))
      t_pixel = (v0t * alpha + v1t * beta + v2t * gamma) * z_pixel

      if (alpha >= 0) and (beta >= 0) and (gamma >= 0):
        if (x >= 0 and x <= 511 and y >= 0 and y <= 511):
          if (z_pixel < z_buffer[x][y]):
            render6.putpixel( (x, y), getColorPhong(v0n, v1n, v2n) )
            z_buffer[x][y] = z_pixel

#AA
for y in range(height):
  for x in range(width):
    RGB1 = np.array([render1.getpixel((x, y))[0], render1.getpixel((x, y))[1], render1.getpixel((x, y))[2]])
    RGB2 = np.array([render2.getpixel((x, y))[0], render2.getpixel((x, y))[1], render2.getpixel((x, y))[2]])
    RGB3 = np.array([render3.getpixel((x, y))[0], render3.getpixel((x, y))[1], render3.getpixel((x, y))[2]])
    RGB4 = np.array([render4.getpixel((x, y))[0], render4.getpixel((x, y))[1], render4.getpixel((x, y))[2]])
    RGB5 = np.array([render5.getpixel((x, y))[0], render5.getpixel((x, y))[1], render5.getpixel((x, y))[2]])
    RGB6 = np.array([render6.getpixel((x, y))[0], render6.getpixel((x, y))[1], render6.getpixel((x, y))[2]])
    resultRGB = RGB1*0.128 + RGB2*0.119 + RGB3*0.294 + RGB4*0.249 + RGB5*0.104 + RGB6*0.106
    render.putpixel( (x, y), (int(resultRGB[0]), int(resultRGB[1]), int(resultRGB[2])))

render
