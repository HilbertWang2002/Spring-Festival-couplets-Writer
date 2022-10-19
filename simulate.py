from robolink import *      # 加载robolink模块
from robodk import *        # 加载robodk模块

import sys                  # 加载sys模块
import os                   # 加载os
import json
import time
import math
import vision.camera as camera
import cv2
import vision.couplet_ocr as couplet_ocr

stroke_count = 0
RDK = Robolink(args='RobotWorkstation.rdk')            # 定义RoboDK工作站
cam = camera.CameraModule(1)
path_stationfile = RDK.getParam('PATH_OPENSTATION')     # 获取当前工作站的路径
sys.path.append(os.path.abspath(path_stationfile))      # 将当前工作站的路径添加到系统路径中
from character.d import pen_track

# 定义工作站中的对象
robot = RDK.Item('UR3')            # 定义机器人对象
write_frame = RDK.Item('坐标系')              # 定义写字坐标系：write_frame
write_tool = RDK.Item('书写工具')                    # 定义写字工具
pixel = RDK.Item('像素点')                        # 定义写字像素点
image_template = RDK.Item('模板')                 # 定义画板的模板
image = RDK.Item('画板')                          # 预定义工作站的画板

robot.setPoseFrame(write_frame)                              # 定义机器人的工件坐标系
robot.setPoseTool(write_tool)                                # 定义机器人的工具坐标系

######## 定义函数 ###########

# 定义函数：将二维空间中的点转化为三维空间中的点（4*4矩阵）
def point2D_2_pose(point, tangent):
    return transl(point.x, point.y, 0)*rotz(tangent.angle())

# s=0，笔从不接触纸面到接触纸面
def point2D_2_pose0(point, tangent):
    return transl(point.x, point.y, )*rotz(tangent.angle())

# s=1，接触保持纸面
def point2D_2_pose1(point, tangent):
    return transl(point.x, point.y, )*rotz(tangent.angle())

# s=2，从接触纸面到离开纸面
def point2D_2_pose2(point, tangent):
    return transl(point.x, point.y, )*rotz(tangent.angle())

def write_robot(list, item_frame, item_tool, robot):
    APPROACH = 78                               # 定义常量APPROACH为100     
    orient_frame2tool = roty(pi)
    global stroke_count
    x = None
    y = None
    for p in list:
        size = 90
        x = p[0]*size/1024
        y = p[1]*size/1024
        #y = p[0]*size/1024
        #x = size - p[1]*size/1024
        if p[3]==0 or p[3]==2:
            target0 = transl(x, y, 20)*orient_frame2tool         # 将p_0转化为机器人目标点target_0(4*4矩阵)
        elif p[3]==1:
            z = 3+(3.8-math.log(p[2]))*2-p[2]*0.07
            if z<0:
                z = 0
            target0=transl(x, y, z)*orient_frame2tool
        target0_app = target0*transl(0,0,-APPROACH)             # 定义目标点：target0_app
        print(str(target0_app))
#        raise Exception
        robot.MoveL(target0_app)
        if p[3]==2:
            stroke_count += 1
            print(stroke_count)
    
def dip(item_frame, item_tool, robot):
    home_joints = [-45,-90,-70,-90,90,0] 
    APPROACH = 125  
    orient_frame2tool = roty(pi)
    robot.MoveJ(home_joints)
    # go down
    x = -150
    y = -200
    target = transl(x, y, -APPROACH)*orient_frame2tool
    robot.MoveL(target)
    
    # rotate
    target1 = transl(x+40, y+50, -APPROACH+20)*orient_frame2tool
    # target1 = target1*rotz(1.0)
    robot.MoveL(target1)
    target = transl(x-40, y-50, -APPROACH+22)*orient_frame2tool
    robot.MoveL(target)
    target1 = transl(x+40, y+50, -APPROACH+24)*orient_frame2tool
    # target1 = target1*rotz(-1.0)
    robot.MoveL(target1)
    target = transl(x-40, y-50, -APPROACH+26)*orient_frame2tool
    robot.MoveL(target)
    target1 = transl(x+40, y+50, -APPROACH+28)*orient_frame2tool
    # target1 = target1*rotz(1.0)
    robot.MoveL(target1)
    time.sleep(2)
    
    # lift
    robot.MoveL(home_joints)
    
    # lift
    #robot.MoveL(home_joints)

    

    # robot.MoveL(home_joints) 

######## 机器人写字主程序 #########


#c = input('请输入对联：')
    
from flask import Flask, request, render_template
import requests
app = Flask(__name__)
robot_busy = False

@app.route('/')
def main():
    return '<p>Hello, World!<p>'
    
@app.route('/test')
def test():
    global robot_busy
    if robot_busy:
        return 'Robot is busy.'
    else:
        robot_busy = True
    couplet = '测试'
    for s in couplet:
        print(s)
        # json_path = os.path.join(path_stationfile,'characters_tracks.json')
        # with open(json_path,'r') as js:
            # track = json.load(js)
        # a = track[s]
        a = pen_track(s)
        a = [j for i in a for j in i]
        write_robot(a, write_frame, write_tool, robot)
    robot_busy = False
    return 'finished'

@app.route('/connect', methods=['GET',])
def connect():
    if RDK.RunMode() != RUNMODE_SIMULATE:
        return 'Already connected.'
    else:
        robot.setConnectionParams('192.168.1.112',2000,'/programs/', 'root','easybot')
        #robot.setConnectionParams('192.168.1.3',2000,)
        success = robot.Connect()
        status, status_msg = robot.ConnectedState()
        time.sleep(2)
        if status != ROBOTCOM_READY:
            # Stop if the connection did not succeed
            return status_msg
        else:
            RDK.setRunMode(RUNMODE_RUN_ROBOT) 
            inititial()
            return success

@app.route('/init')
def inititial():
    home_joints = [-45,-90,-70,-90,90,0]                                # 定义机器人的起始位置
  # home_joints = [0,0,15,-90,90,0]     
    robot.MoveJ(home_joints)
    return 'Success'
            
@app.route('/write/<c>')
def write(c):
    global robot_busy
    global stroke_count
    if robot_busy:
        return 'Robot is busy.'
    else:
        robot_busy = True
    dip1()
    for s in c:
        print('write',stroke_count)
        if stroke_count>=15:
            stroke_count = 0
            dip1()
        if s == " ":
            time.sleep(10)
            continue
        print(s)
        a = pen_track(s)
        if a is None:
            continue
        a = [j for i in a for j in i]
        write_robot(a, write_frame, write_tool, robot)
        time.sleep(2)
    robot_busy = False
    return 'finished'

@app.route('/couplet', methods=['GET'])
def couplet():
    data = request.values.to_dict()['c']
    r = requests.get(f'http://localhost:12345/{data}')
    if r.status_code == 200 and r.text != 'Failed':
        if write(data)=='Robot is busy.':
            return 'Robot is busy.'
        inititial()
        time.sleep(10)
        if write(r.text)=='Robot is busy.':
            return 'Robot is busy.'
    return finished

@app.route('/dip')
def dip1():
    print('dip')
    dip(write_frame, write_tool, robot)
    return '1'
    

@app.route('/ocr')
def ocr():
    global cam
    winName = 'image'
    cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)
    start_time = time.time()
    while(1):
        ret, frame = cam.cam.read()
        if not ret:
            break
        cv2.imshow(winName , frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time()-start_time >= 5:
            break
    s = couplet_ocr.ocr_current_camera(cam)
    print(s)
    r = requests.get(f'http://localhost:12345/{s}')
    if r.status_code == 200 and r.text != 'Failed':
        result = r.text
        return render_template('./tem.html', s=s+' '+result)
    return r.text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)