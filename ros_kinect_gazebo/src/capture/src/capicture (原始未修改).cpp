/*
 * open_and_contours.cpp
 *
 *  Created on: Mar 23, 2017
 *      Author: exbot
 */

// Includes all the headers necessary to use the most common public pieces of
// the ROS system.
#include <ros/ros.h>
// Use image_transport for publishing and subscribing to images in ROS
#include <image_transport/image_transport.h>
// Use cv_bridge to convert between ROS and OpenCV Image formats
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/image_encodings.h>
// Include headers for OpenCV Image processing
#include <opencv2/imgproc/imgproc.hpp>
// Include headers for OpenCV GUI handling
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <sstream>
using namespace cv;
using namespace std;

// Store all constants for image encodings in the enc namespace to be used
// later.
namespace enc = sensor_msgs::image_encodings;
void image_socket(Mat inImg);

Mat img_binary;
Mat img_gray;
Mat image_roi;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
vector<Point> contour;

// This function is called everytime a new image is published
void imageCallback(const sensor_msgs::ImageConstPtr &original_image) {
  // Convert from the ROS image message to a CvImage suitable for working with
  // OpenCV for processing
  cv_bridge::CvImagePtr cv_ptr;
  try {
    // Always copy, returning a mutable CvImage
    // OpenCV expects color images to use BGR channel order.
    cv_ptr = cv_bridge::toCvCopy(original_image, enc::BGR8);
  }
  catch (cv_bridge::Exception &e) {
    // if there is an error during conversion, display it
    ROS_ERROR("tutorialROSOpenCV::main.cpp::cv_bridge exception: %s", e.what());
    return;
  }
  image_socket(cv_ptr->image);
}

void image_socket(Mat inImg) {

  if (inImg.empty()) {
    ROS_INFO("Camera image empty");
    return; // break;
  }

  cvtColor(inImg, img_gray, CV_BGR2GRAY);
  threshold(img_gray, img_binary, 130, 255, CV_THRESH_BINARY);
  findContours(img_binary, contours, hierarchy, CV_RETR_CCOMP,
               CV_CHAIN_APPROX_SIMPLE);
  // drawContours(inImg, contours, -1, Scalar(0, 255, 0), 2);
  int m = contours.size(); //得到轮廓的数量

  int n = 0;

  for (int i = 0; i < m; ++i) {
    n = contours[i].size();
    for (int j = 0; j < n; ++j) {
      contour.push_back(contours[i][j]); //读取每个轮廓的点
    }
    double area = contourArea(contour); //取得轮廓面积

    if (area > 40000) //只画出轮廓大于1000的点
    {

      Rect rect = boundingRect(contour); //轮廓区域按矩形截
      int x = rect.x - 10;
      int y = rect.y - 10;
      int height = rect.height + 20;
      int width = rect.width + 20;

      if (x > 15 && y > 15 && (x + width + 15) < inImg.size().width &&
          (y + height + 15) < inImg.size().height) {
        //画矩形和保存截取的图片完全是两回事
        //把矩形截取的区域扩大10个像素点进行保存
        int x = rect.x - 10;
        int y = rect.y - 10;
        int height = rect.height + 20;
        int width = rect.width + 20;
        Rect rect_roi(x, y, width, height);
        image_roi = inImg(rect_roi); //截取矩形区
        imwrite("/home/exbot/ros_kinect_gazebo/src/picture/test.png",
                image_roi); //保存截取图片
      }
      //这里才是画出矩形框，方便从view里观看
      rectangle(inImg, rect, Scalar(0, 0, 255), 3);
    }

    contour.clear();
  }
  namedWindow("Kinect image", CV_WINDOW_NORMAL);
  imshow("Kinect image", inImg); //显示图片
  char c = cvWaitKey(33);        //暂停33ms

  if (c == 27) {
    ROS_INFO("Exit boss");
  }
}

/**
 * This is ROS node to track the destination image
 */
int main(int argc, char **argv) {
  ros::init(argc, argv, "Kinect_image");
  ROS_INFO("-------Please wait!---------");

  ros::NodeHandle nh;

  image_transport::ImageTransport it(nh);

  image_transport::Subscriber sub =
      it.subscribe("camera/rgb/image_raw", 1, imageCallback);

  ros::spin();

  // ROS_INFO is the replacement for printf/cout.
  ROS_INFO("tutorialROSOpenCV::main.cpp::No error.");
}
