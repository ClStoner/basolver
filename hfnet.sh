echo "clear terminal ..."

clear

# echo "enter dataset dir ..."
# cd /root/share/datasets/rgbd

# echo "rosbag play dataset ..."
# rosbag play indoor2.bag & 

echo "starting hfnet ros node..."
roslaunch hfnet hfnet.launch