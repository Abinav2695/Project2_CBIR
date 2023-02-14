
/* Abinav Anantharaman
   CS 5330 Spring 2023
   Image Display code
   Source file
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "filter.h"


int h_slider_low = 0;
int s_slider_low = 0;
int v_slider_low = 0;
int h_slider_high = 0;
int s_slider_high = 0;
int v_slider_high = 0;

cv::Mat image_to_be_displayed;
cv::Mat hsv_image;
cv::Mat hsv_thresh_image;
cv::Mat hsv_mask;
/**
 * Trackbar callback function
 * This function is called everytime the position of the brightness and contrast trackbars are changed
 * @param int --> value of the slider position (0-max value)
 * @param void* --> user data to pass to callback function.
 * @return no return values .
 */  
void SliderCallback(int,void *userdata )
{   
    hsv_thresh_image = cv::Mat::zeros(image_to_be_displayed.size(),CV_8UC3);
    cv::cvtColor(image_to_be_displayed,hsv_image,cv::COLOR_BGR2HSV);
    cv::inRange(hsv_image,cv::Scalar(h_slider_low,s_slider_low,v_slider_low),cv::Scalar(h_slider_high,s_slider_high,v_slider_high),hsv_mask);
    cv::bitwise_and(image_to_be_displayed,image_to_be_displayed,hsv_thresh_image,hsv_mask);
    cv::imshow("Display window",hsv_thresh_image);
    // std::cout << hsv_mask.channels() << " " << hsv_mask.size() << " " <<hsv_mask.type() << std::endl;
}


int main(int argc,char* argv[])
{
    if (argc>=2)
    {
        std::string image_path = argv[1];
        std::cout << "Image path: " << image_path << std::endl;

        image_to_be_displayed = cv::imread(image_path, cv::IMREAD_COLOR);

        if(image_to_be_displayed.empty())  //if no image found or error in reading image , exit code
        {
            std::cout << "Could not read the image: " << image_path << std::endl;
            return 0;
        }

        cv::imshow("Display window", image_to_be_displayed);  //opencv image window
        bool keep_looping = true;
        
        while(keep_looping)
        {
            char key_pressed = cv::waitKey(0);
            // std::cout << char(key_pressed) << std::endl;

            switch(key_pressed)
            {
                case 'g': //bgr to gray conversion
                {
                    cv::Mat gray_image;
                    cv::cvtColor(image_to_be_displayed,gray_image,cv::COLOR_BGR2GRAY);
                    cv::imshow("Display window", gray_image);
                    break;
                }

                case 'r':  //bgr to rgb conversion
                {
                    cv::Mat rgb_image;
                    cv::cvtColor(image_to_be_displayed,rgb_image,cv::COLOR_BGR2RGB);
                    cv::imshow("Display window", rgb_image);
                    break;
                }

                case 'h': //bgr to hsv conversion command
                {
                    
                    cv::namedWindow("trackbar_window");
                    //Create track bar to change hue
                    cv::createTrackbar("Hue_low", "trackbar_window", &h_slider_low, 180, SliderCallback);
                    //Create track bar to change saturation
                    cv::createTrackbar("Sat_low", "trackbar_window", &s_slider_low, 255, SliderCallback);
                     //Create track bar to change value
                    cv::createTrackbar("Val_low", "trackbar_window", &v_slider_low, 255, SliderCallback);
                    //Create track bar to change hue
                    cv::createTrackbar("Hue_high", "trackbar_window", &h_slider_high, 180, SliderCallback);
                    //Create track bar to change saturation
                    cv::createTrackbar("Sat_high", "trackbar_window", &s_slider_high, 255, SliderCallback);
                     //Create track bar to change value
                    cv::createTrackbar("Val_high", "trackbar_window", &v_slider_high, 255, SliderCallback);

                    //Initialise trackbar to mid values
                    SliderCallback(h_slider_low,0);   
                    SliderCallback(s_slider_low, 0);
                    SliderCallback(v_slider_low, 0);
                    SliderCallback(h_slider_high,0);   
                    SliderCallback(s_slider_high, 0);
                    SliderCallback(v_slider_high, 0);

                    break;
                }

                case 'o': //original image
                {
                    cv::imshow("Display window", image_to_be_displayed);
                    break;
                }

                case 'b':  //image blur command
                {
                    cv::Mat blur_image;
                    // cv::GaussianBlur(image_to_be_displayed, blur_image, cv::Size(5, 5), 0);
                    blur5x5(image_to_be_displayed, blur_image);
                    cv::imshow("Display window", blur_image);
                    break;
                }

                case 'q':  //quit command
                {   
                    std::cout<< "Shutting down the application!" << std::endl;
                    keep_looping = false;
                    break;
                } 
                default:
                    std::cout<< "~~invalid input~~" << std::endl;
                    break;
            }
        }

        cv::destroyAllWindows();  //destroy all open windows
    }

    else{
        std::cout << "Please specify Image Path!!!" << std::endl;
        std::cout << "Usage : " << argv[0] << " {full image path}" <<std::endl;
    }
    
    return 0;
}
