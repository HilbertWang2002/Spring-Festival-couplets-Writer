filename = '矢量图-1颜色.svg'

with open(filename, 'r') as f:
    data = f.readlines()
    
svg_template = '''<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Created with svg_stack (http://github.com/astraw/svg_stack) -->
<svg xmlns="http://www.w3.org/2000/svg" xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" version="1.1" width="1756.25" height="1198.75">
  <defs/>
  <g id="id0:id0" transform="matrix(1.25,0,0,1.25,0.0,0.0)"><g transform="translate(0.000000,959.000000) scale(0.100000,-0.100000)" fill="#000000" stroke="none">
    ...
</g>
</g>
</svg>

'''
count = 0
for i in data:
    if i.startswith('<path'):
        with open(f's{count}.svg', 'w') as f:
            f.write(svg_template.replace('...', i))
        count += 1