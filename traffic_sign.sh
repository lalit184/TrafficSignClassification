# gnome-terminal -e command
# x-terminal-emulator
echo Hello, this script runs roscore, yolo, plays the bag file as a loop 
echo performs fisheye corrector and executes execute.py script
echo only works when the file is saved with execute.py
echo Please enter the entire bagfile name
echo .................................................................
echo these are the current bagfiles
echo .................................................................
echo 20_30_40_50.bag
echo 60_70_80_90.bag
echo compahead_100_height_nostop.bag
echo comprightstraight_cycletrack_lrightcurve_leftcurve.bag
echo keepleft_comp-leftahead_rightahead_leftstraight.bag
echo no-_ahead_left_right_u.bag



read -p 'bagfilename: ' file_name

yolo()
{
	sleep 3s
	cd $HOME/darknet_ws 
	source devel/setup.bash 
	roslaunch darknet_ros darknet_rosv3_sign.launch
}

fisheye()
{
	sleep 3s
	cd $HOME/final_lane 
	source devel/setup.bash 
	rosrun vatsal Lane_fused
}

bagplay()
{	

	cd ../../../bagfiles/Tsigns 
	rosbag play -l $file_name
}

roscore & fisheye & bagplay & yolo

trap killall -9 roscore,rosmaster
