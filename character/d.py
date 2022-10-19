import cairosvg
import json
import os
import imageio
import numpy as np
import skimage
import math
import matplotlib.pyplot as plt

svg_temp = 'temp/temp.svg'
png_temp = 'temp/temp.png'
show_every_step = False

os.makedirs('temp', exist_ok=True)

with open(os.path.join(os.path.dirname(__file__), 'graphics.txt'), 'r', encoding='utf-8') as f:
	data = f.readlines()
aa = []
for i in data:
    aa.append(json.loads(i[:-1]))
data = aa

svg_template = '''
<svg width="1024" height="1024" xmlns="http://www.w3.org/2000/svg">
 <g transform="scale(1, -1) translate(0, -900)">
  <title>Layer 1</title>
  ...
  </g>
</svg>
'''

def find_n_points(image, n, num=1):
    
    points = []
    for _y in range(1, image.shape[1]-1):
        for _x in range(1, image.shape[0]-1):
            if image[_x, _y] == num:
                if np.sum(image[_x-1:_x+2, _y-1:_y+2]) == num*(n+1):
                    points.append((_x, _y))
    return points

def find_next(image, start_point, ignored_points = []):
    _x = start_point[0]
    _y = start_point[1]
    image[_x, _y] = 0
    near = []
    for i in range(_x-1, _x+2):
        for j in range(_y-1, _y+2):
            if i != _x or j != _y:
                if image[i, j] == 1 and (i,j) not in ignored_points:
                    near.append((i, j))
    return near

def simplify(image, start_point):
    image_copy = image.copy()
    return _simplify(image_copy, start_point)
    
    
    
    
def _simplify(image, start_point, ignored_points=[]):
    #print('calling _simplify() with arguments:', start_point, ignored_points)
    _next_x = _x = start_point[0]
    _next_y = _y = start_point[1]
    points = [(_x, _y)]
    last_delta_x = 0
    last_delta_y = 0
    length = 1
    while(True):
        length += 1
        next_points = find_next(image, (_next_x, _next_y), ignored_points)
        if len(next_points) > 1:
            p = []
            m = 0
            for i in next_points:
                a, b = _simplify(image, i, ignored_points=next_points)
                if b > m:
                    m = b
                    p = a
            length += m
            points += p
            break
        elif len(next_points) == 0:
            points.append((_next_x, _next_y))
            break
        else:
            _next_x = next_points[0][0]
            _next_y = next_points[0][1]
            if (_next_x-_x)**2+(_next_y-_y)**2 <= 900:#distance
                continue
            else:
                this_delta_x = _next_x - _x
                this_delta_y = _next_y - _y
                if last_delta_x == 0 and last_delta_y == 0:
                    _x, _y, last_delta_x, last_delta_y = _next_x, _next_y, this_delta_x, this_delta_y
                    points.append((_x, _y))
                    continue
                T = (this_delta_x*last_delta_x+this_delta_y*last_delta_y)/(this_delta_x**2+this_delta_y**2)**0.5/(last_delta_x**2+last_delta_y**2)**0.5
                if T>=1:#cosine_similarity
                    continue
                _x, _y, last_delta_x, last_delta_y = _next_x, _next_y, this_delta_x, this_delta_y
                points.append((_x, _y))
    #print('_simplify() is returning:', length, points)
    return points, length
        

def pen_track(c):
    try:
        os.mkdir('temp')
    except:
        pass
    #strokes_png = os.listdir(f'out/{c}')
    for i in data:
        if c == i['character']:
            strokes_path = i['strokes']
    if strokes_path is None:
        return

    tracks = []
    for i in strokes_path:
        a = '\n'
        b = '<path d="..."></path>\n'
        _temp = b.replace('...', i)
        a += _temp
        with open(svg_temp, 'w') as f:
            f.write(svg_template.replace('...', a))
        cairosvg.svg2png(url=svg_temp, write_to=png_temp)
        _temp_full = imageio.imread(png_temp)
        imageio.imwrite(png_temp, np.where(skimage.morphology.skeletonize(np.where(np.max(_temp_full, axis=-1)>0, 1, 0))==True, 1, 0))
        _temp_full = np.where(np.max(_temp_full, axis=-1)==255, 1, 0)
        print(_temp_full.shape)
        print(np.max(_temp_full))
    
            
    
        _temp = imageio.imread(png_temp)
        _temp = np.where(_temp==255, 1, 0)
        points_1 = find_n_points(_temp, 1)
        '''
        points_3 = find_n_points(_temp, 3)
        points_4 = find_n_points(_temp, 4)
        for j in points_4:
            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:
                    if (j[0]+k, j[1]+l) in points_3:
                        points_3.remove((j[0]+k, j[1]+l))
        for j in range(len(points_3)):
            if j+2<len(points_3):
                if abs(points_3[j][0]-points_3[j+1][0]) <= 1 and abs(points_3[j][1]-points_3[j+1][1]) <= 1 and abs(points_3[j+2][0]-points_3[j+1][0]) <= 1 and abs(points_3[j+2][1]-points_3[j+1][1]) <= 1:
                   points_3.pop(j)
                   points_3.pop(j+1)
        points_joint = points_3 + points_4 
        '''
        m = 2049
        for j in points_1:
            if (j[0]+j[1]) < m:
                m = (j[0]+j[1])
                start_point = j
        a, b = simplify(_temp, start_point)
        
        width = []
        for j in range(len(a)):
            if j == 0:
                _tan = (a[1][0] - a[0][0], a[1][1] - a[0][1])
            elif j == len(a)-1:
                _tan = (a[j][0] - a[j-1][0], a[j][1] - a[j-1][1])
            else:
                _tan = (a[j+1][0] - a[j-1][0], a[j+1][1] - a[j-1][1])
            _angle = math.atan(_tan[1]/(_tan[0]+1e-6))+math.pi/2
            _delta_x = math.cos(_angle)
            _delta_y = math.sin(_angle)
            k = 1
            while(True):
                if(_temp_full[int(a[j][0]+k*_delta_x), int(a[j][1]+k*_delta_y)]==0):
                    break
                k += 1
            width.append([k])
            
        a = np.hstack([a, width, [[0]]+[[1]]*(len(a)-2)+[[2]]]).tolist()
        tracks.append(a)
        if show_every_step:
            plt.scatter([z[1] for z in a], [1024-z[0] for z in a], c=[z[2] for z in a])
            plt.xlim((0, 1024))
            plt.ylim((0, 1024))
            plt.show()
    return tracks