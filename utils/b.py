import imageio
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import multiprocessing as mp
import tqdm
import time

show_every_step = False

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
    strokes_png = os.listdir(f'out/{c}')
    tracks = []
    for i in strokes_png:
        _temp = imageio.imread(f'out/{c}/{i}')
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
        a = np.hstack([a, [[0]]+[[1]]*(len(a)-2)+[[2]]]).tolist()
        tracks.append(a)
        if show_every_step:
            plt.scatter([z[1] for z in a], [1024-z[0] for z in a], c=[z[2] for z in a])
            plt.xlim((0, 1024))
            plt.ylim((0, 1024))
            plt.show()
    return tracks
        
def worker(q, return_dict,):
    while(True):
        c = q.get()
        #print(f'{os.getpid()} Got {c}.')
        if c=='q':
            #print(f'{os.getpid()} finished.')
            return
        
        a = pen_track(c)
        a = [j for i in a for j in i]
        return_dict[c] = a
        
        
        
if __name__ == '__main__':
    '''
    q = mp.Queue()
    p = []
    n = 16
    manager = mp.Manager()
    return_dict = manager.dict()
    characters = os.listdir(f'out')
    total = len(characters)
    pbar = tqdm.tqdm(total=total)
    for i in range(n):
        p.append(mp.Process(target=worker, args=(q, return_dict,)))
    for c in characters:
        q.put(c)
    for i in range(n):
        p[i].start()
    for i in range(16):
        q.put('q')
    last_finished = 0
    while(True):
        this_finished = total+16-q.qsize()
        pbar.update(this_finished-last_finished)
        last_finished = this_finished
        time.sleep(1)
        if(q.empty()):
            break
    for i in p:
        i.join()
    json.dump(return_dict, open(r'characters_tracks.json', 'w'))
    '''
    '''
        a = pen_track(c)
        a = [j for i in a for j in i]
        c_dict[c] = a
        print(f'Finished: {c}')
    json.dump(c_dict, open(r'characters_tracks.json', 'w'))
    '''

    s = input('??????????????????')
    for c in range(len(s)):
        a = pen_track(s[c])
        
        a = [j for i in a for j in i]
        print(a)
        plt.subplot(len(s), 1, c+1)
        plt.scatter([z[1] for z in a], [1024-z[0] for z in a], c=[z[2] for z in a])
        plt.xlim((0, 1024))
        plt.ylim((0, 1024))
    plt.show()
    