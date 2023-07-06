# 在VINS-Mono/Fusion框架下，针对滑动窗口的局部BA优化，提供高效求解

开发的BaSolver求解器代码放置与basolver文件夹下，并将求解器引用于VINS/Fusion的滑动窗口优化中，替换Ceres求解器。

## 环境依赖
1. ROS
2. OpenCV with Contrib 3.4.15
3. Eigen 3.3.x
4. Ceres 1.14.x
5. PCL

## CMake
算法中相关实时性以及精度宏开关
```
add_definitions(-DTEST_PERF)
```

## 编译
```
catkin build
```

## 运行程序
`slam.sh`脚本中`euroc_mono.launch`用于单目启动，`euroc_stereo.launch`用于双目启动
```
16    echo "starting slam ros node..."
17    roslaunch launch/euroc_stereo.launch
18    # roslaunch launch/euroc_mono.launch
```
运行
```
roscore
./slam.sh
rosbag play MH_01_easy.bag  
```

## 可视化
可视化可以单独起一个终端，参数为`rviz`配置文件路径，为`config`文件夹中`test_rgbd.rviz`
```
rviz -d config/test_rgbd.rviz
```

