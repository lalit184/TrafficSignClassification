
//ros libraries
#include "ros/ros.h"
//c++ libraries
#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <fstream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/CheckForObjectsAction.h>
#include <utility>


#include "darknet_ros/Array_images.h"
#include "sensor_msgs/Image.h"

#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <algorithm>
#include "std_msgs/Int32MultiArray.h"
#include "std_msgs/Int8.h"
#include <vector>
using namespace cv;
using namespace std;


namespace enc = sensor_msgs::image_encodings;

ros::Subscriber sub_trafficdata;

Mat white;

//Use method of ImageTransport to create image publisher
image_transport::Publisher pub_right;
ros::Publisher pub;

darknet_ros::Array_images rosimgs;
int num = 1;
cv_bridge::CvImagePtr cv_ptr;
//darknet_ros_msgs::BoundingBoxes msg1;
Mat src,dst,input;

std_msgs::Int32MultiArray traffic_direction;


void leftimage(const sensor_msgs::ImageConstPtr& original_image)
{
  
    try
    {
        cv_ptr = cv_bridge::toCvCopy(original_image, enc::BGR8);
 	}
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("videofeed::igvc_IPM.cpp::cv_bridge exception: %s", e.what());
        cout<<"no message received ......................."<<endl;
        return;
    }
    src = cv_ptr->image;
  //imshow("name",dst);
  // cv_ptr->image = dst;
  // pub_right.publish(cv_ptr->toImageMsg());
    
}

void callback_left( const darknet_ros_msgs::BoundingBoxes& msg)
{
  
  darknet_ros_msgs::BoundingBox b;
  input = src.clone();
  //imshow("input",input);
  num = (msg.bounding_boxes).size();
  Mat temp;
  int xmin,ymin,xmax,ymax; 
  Mat signs[num];
  Mat tem[num];
  Mat stitch;

  cout<<"no of detections = "<<num<<endl;
  for(int i =0;i<num;i++)
  {
    b = msg.bounding_boxes[i];

    if (b.Class == "sign")
    {
      xmin = b.xmin;
      ymin = b.ymin;
      xmax = b.xmax;
      ymax = b.ymax;
      temp = input(cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin));
      resize(temp,temp,Size(200,200));
      signs[i] = temp.clone();
    }



  }
/*  tem[0] = signs[0];
  for (int i = 0; i < num-1; ++i)
  {
    cv::hconcat(signs[i+1], tem[i], tem[i+1]);
  }
  
  stitch = tem[num-1];
  stitch.at<Vec3b>(0,0)[0] = num;
*/
//  cout<<"no. of signs detected = "<<num<<endl;
  
  //dst = stitch.clone(); 
    //publishing here
  
  //rosimgs.data.clear();
  // header of rosimgs ignored for now
  for (uint i=0; i<num; i++) {
      try {
          cv_ptr->image = signs[i];
          imshow("detectedsign",signs[0]);
          sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", signs[0]).toImageMsg();
          // pub_right.publish(msg);

          rosimgs.data.push_back(*(cv_ptr->toImageMsg()));
      } catch (cv_bridge::Exception& e) {
          ROS_ERROR("error");
      }
  }
  pub.publish(rosimgs);
  waitKey(1);

}

void callback_check(const std_msgs::Int8& msg)
{
	int data = msg.data;
	if(data==0)
	{
		cout<<"no object found"<<endl;
		dst = white.clone();
	}
}
int main(int argc, char **argv)
{	
  ros::init(argc, argv, "sign");

  ros::NodeHandle nh;

  image_transport::ImageTransport it(nh);
  cout<<"reached.........."<<endl;
  white = imread("white.png",1);
  cout<<"loaded..."<<endl;
  //resize(white,white,Size(200,200));

	image_transport::Subscriber sub_left = it.subscribe("/camera2/fisheye",1, leftimage);
	
	ros::Subscriber roi_left = nh.subscribe("/darknet_rosign/bounding_boxes", 1, callback_left);
	
	ros::Subscriber sub_check = nh.subscribe("/darknet_rosign/found_object",1,callback_check); 
  	
  // pub_right = it.advertise("/trafficsignimage", 1);
  // pub = nh.advertise<darknet_ros::Array_images>("/rois", 1000, true);
	pub = nh.advertise<darknet_ros::Array_images>("/trafficsignimage", 1000, true);

	
	ros::Rate loop_rate(10);

	while(ros::ok())
	{	
		ros::spinOnce();
		loop_rate.sleep();
	}

	ROS_INFO("videofeed::occupancygrid.cpp::No error.");

}