import cairosvg
import json
import os
import imageio
import numpy as np
import skimage

svg_temp = 'temp.svg'
png_temp = 'temp.png'

with open('graphics.txt', 'r', encoding='utf-8') as f:
	data = f.readlines()
aa = []
for i in data:
    aa.append(json.loads(i[:-1]))

svg_template = '''
<svg width="1024" height="1024" xmlns="http://www.w3.org/2000/svg">
 <g transform="scale(1, -1) translate(0, -900)">
  <title>Layer 1</title>
  ...
  </g>
</svg>
'''
    
    
if __name__ == '__main__':
    try:
        os.mkdir('out1')
    except:
        pass
    for c in aa[1000:1005]:
        character_name = c['character']
        strokes_raw = c['strokes']
        try:
            os.mkdir('out1\\'+character_name)
        except Exception as e:
            print(e)
            continue
        a = '\n'
        b = '<path d="..."></path>\n'
        for i in range(len(strokes_raw)):
            _temp = b.replace('...', strokes_raw[i])
            a += _temp
            with open(svg_temp, 'w') as f:
                f.write(svg_template.replace('...', a))
            cairosvg.svg2png(url=svg_temp, write_to=png_temp)
            imageio.imwrite(f'out1\\{character_name}\\{i}.png', np.where(skimage.morphology.skeletonize(np.where(np.max(imageio.imread(png_temp), axis=-1)>0, 1, 0))==True, 1, 0))