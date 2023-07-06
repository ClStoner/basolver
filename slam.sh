echo "clear terminal ..."

# clear

echo "clear result img ..."
rm -rf results/log/*.*


echo "starting slam ros node..."
# roslaunch launch/euroc_stereo.launch
roslaunch launch/euroc_mono.launch
