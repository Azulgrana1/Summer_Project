// main.cpp : Defines the entry point for the console application.  
//  
 
#include <iostream>  
#include <sstream>
#include <string>
#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "SerialPort.h"

using namespace cv;
using namespace std;
//initial min and max HSV filter values.
//these will be changed using trackbars
//red threshold B:0-50 G:0-50 R:100-255
//green threshold B:0-50 G:100-255 R:0-50
Scalar RED_MIN(0,0,100);
Scalar RED_MAX(50,50,255);
Scalar GREEN_MIN(0,100,0);
Scalar GREEN_MAX(150,255,60);

//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS=50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;
//names that will appear at the top of each window
const string windowName = "Original Image";
const string windowName1 = "Red Thresholded Image";
const string windowName2 = "Green Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

string intToString(int number){
    std::stringstream ss;
    ss << number;
    return ss.str();
}

void drawObject(int x, int y,Mat &frame){

    //use some of the openCV drawing functions to draw crosshairs
    //on your tracked image!

    //UPDATE:JUNE 18TH, 2013
    //added 'if' and 'else' statements to prevent
    //memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

    circle(frame,Point(x,y),20,Scalar(0,255,0),2);
    if(y-25>0)
        line(frame,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,0),Scalar(0,255,0),2);
    if(y+25<FRAME_HEIGHT)
        line(frame,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(x,FRAME_HEIGHT),Scalar(0,255,0),2);
    if(x-25>0)
        line(frame,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(0,y),Scalar(0,255,0),2);
    if(x+25<FRAME_WIDTH)
        line(frame,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);
    else line(frame,Point(x,y),Point(FRAME_WIDTH,y),Scalar(0,255,0),2);

    putText(frame,intToString(x)+","+intToString(y),Point(x,y+30),1,1,Scalar(0,255,0),2);

}
void morphOps(Mat &thresh){

//create structuring element that will be used to "dilate" and "erode" image.
//the element chosen here is a 3px by 3px rectangle

    Mat erodeElement = getStructuringElement( MORPH_RECT,Size(3,3));
//dilate with larger element so make sure object is nicely visible
    Mat dilateElement = getStructuringElement( MORPH_RECT,Size(8,8));

    erode(thresh,thresh,erodeElement);
    erode(thresh,thresh,erodeElement);

    dilate(thresh,thresh,dilateElement);
    dilate(thresh,thresh,dilateElement);
	
}

bool trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed)
{

    Mat temp;
    threshold.copyTo(temp);
//these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
//find contours of filtered image using openCV findContours function
    findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
//use moments method to find our filtered object
    double refArea = 0;
    bool objectFound = false;
    if (hierarchy.size() > 0) 
    {
        int numObjects = hierarchy.size();
        //if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
        if(numObjects<MAX_NUM_OBJECTS)
        {
            for (int index = 0; index >= 0; index = hierarchy[index][0]) 
            {

                Moments moment = moments((cv::Mat)contours[index]);
                double area = moment.m00;

                //if the area is less than 20 px by 20px then it is probably just noise
                //if the area is the same as the 3/2 of the image size, probably just a bad filter
                //we only want the object with the largest area so we safe a reference area each
                //iteration and compare it to the area in the next iteration.
                if(area>MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea)
                {
                    x = moment.m10/area;
                    y = moment.m01/area;
                    objectFound = true;
                    refArea = area;
                }
                else 
                    objectFound = false;

            }
			//let user know you found an object
            if(objectFound ==true)
            {
                putText(cameraFeed,"Tracking Object",Point(0,50),2,1,Scalar(0,255,0),2);
                //draw object location on screen
                drawObject(x,y,cameraFeed);
            }
        }
        else 
            putText(cameraFeed,"TOO MUCH NOISE! ADJUST FILTER",Point(0,50),1,2,Scalar(0,0,255),2);
    }
    return objectFound;
}

int main(int argc, char* argv[])  
{   
    //Setting up serial port
    int fd;
    char *dev = "/dev/ttyUSB0";
    fd = OpenDev(dev);
    if (fd>0)
   	set_speed(fd,9600);
    else
    {
        printf("Can't Open Serial Port!\n");
        exit(0);
    }
    if (set_Parity(fd,8,1,'N')== 0)
    {
        printf("Set Parity Error\n");
        exit(1);
    }
   
    //some boolean variables for different functionality within this
    //program
    bool trackObjects = true;
    bool useMorphOps = true;
    //Matrix to store each frame of the webcam feed
    Mat cameraFeed;
    //matrix storage for binary threshold image
    Mat red_threshold;
    Mat green_threshold;
    //x and y values for the location of the object
    int x=0, y=0;
    //video capture object to acquire webcam feed
    VideoCapture capture;
    //open capture object at location zero (default location for webcam)
    capture.open(1);
    //set height and width of capture frame
    capture.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);
    //start an infinite loop where webcam feed is copied to cameraFeed matrix
    //all of our operations will be performed within this loop

    bool red_on, green_on;
    while(1){
        //store image to matrix
        capture.read(cameraFeed);
        //filter BGR image between values and store filtered image to
        //threshold matrix
        inRange(cameraFeed,RED_MIN,RED_MAX,red_threshold);
        inRange(cameraFeed,GREEN_MIN,GREEN_MAX,green_threshold);
        //perform morphological operations on thresholded image to eliminate noise
        //and emphasize the filtered object(s)
        if(useMorphOps)
        {
            morphOps(red_threshold);
            morphOps(green_threshold);
        }
        //pass in thresholded frame to our object tracking function
        //this function will return the x and y coordinates of the
        //filtered object
        if(trackObjects)
        {
            red_on = trackFilteredObject(x,y,red_threshold,cameraFeed);
            green_on = trackFilteredObject(x,y,green_threshold,cameraFeed);
        }

        char status[1];
        if(!red_on && !green_on)
            status[0] = 0;
        else if(!red_on && green_on)
            status[0] = 1;
        else if(red_on && !green_on)
            status[0] = 2;
        else //(red_on && green_on)
            status[0] = 3;
        write(fd, status, 1);

        //show frames 
        imshow(windowName1,red_threshold);
        imshow(windowName2,green_threshold);
        imshow(windowName,cameraFeed);
        //imshow(windowName1,HSV);
		
        //delay 30ms so that screen can refresh.
        //image will not appear without this waitKey() command
        waitKey(10);
    }

    return 0; 
}  
