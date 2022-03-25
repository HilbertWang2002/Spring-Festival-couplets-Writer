from robolink import *      # 加载robolink模块
from robodk import *        # 加载robodk模块
from b import *

import sys                  # 加载sys模块
import os                   # 加载os
path_stationfile = RDK.getParam('PATH_OPENSTATION')     # 获取当前工作站的路径
sys.path.append(os.path.abspath(path_stationfile))      # 将当前工作站的路径添加到系统路径中

RDK = Robolink()            # 定义RoboDK工作站

# 定义工作站中的对象
robot = RDK.Item('UR3')            # 定义机器人对象
write_frame = RDK.Item('绘图坐标系')              # 定义写字坐标系：write_frame
write_tool = RDK.Item('书写工具')                    # 定义写字工具
pixel = RDK.Item('像素点')                        # 定义写字像素点
image_template = RDK.Item('模板')                 # 定义画板的模板
image = RDK.Item('画板')                          # 预定义工作站的画板

######## 定义函数 ###########

# 定义函数：将二维空间中的点转化为三维空间中的点（4*4矩阵）
def point2D_2_pose(point, tangent):
    return transl(point.x, point.y, 0)*rotz(tangent.angle())

# s=0，笔从不接触纸面到接触纸面
def point2D_2_pose0(point, tangent):
    return transl(point.x, point.y, 10)*rotz(tangent.angle())

# s=1，接触保持纸面
def point2D_2_pose1(point, tangent):
    return transl(point.x, point.y, 0)*rotz(tangent.angle())

# s=2，从接触纸面到离开纸面
def point2D_2_pose2(point, tangent):
    return transl(point.x, point.y, 10)*rotz(tangent.angle())

def write_robot(list, item_frame, item_tool, robot):
    APPROACH = 150                                              # 定义常量APPROACH为100     
    home_joints = [-45,-90,-90,-90,90,0]                                # 定义机器人的起始位置
    
    robot.setPoseFrame(item_frame)                              # 定义机器人的工件坐标系
    robot.setPoseTool(item_tool)                                # 定义机器人的工具坐标系
    robot.MoveJ(home_joints)                                    # 机器人移动到起始位置
    
    orient_frame2tool = roty(pi) 

    for p in list:
        if p[2]==0 or p[2]==2:
            target0 = transl(p[0], p[1], 10)*orient_frame2tool         # 将p_0转化为机器人目标点target_0(4*4矩阵)
        elif p[2]==1:
            target0 = transl(p[0], p[1], 0)*orient_frame2tool
        target0_app = target0*transl(0,0,-APPROACH)             # 定义目标点：target0_app
        robot.MoveL(target0_app)    

    robot.MoveL(home_joints) 

######## 机器人写字主程序 #########
s = input('请输入汉字：')
for c in range(len(s)):
    a = pen_track(s[c])
    a = [j for i in a for j in i]
write_robot(a, write_frame, write_tool, robot)

