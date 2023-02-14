
/* Abinav Anantharaman
   CS 5330 Spring 2023
   Filter library
   Source file
*/

#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "filter.h"


/**
 * Function to convert a BGR color image into grayscale
 * The function calculates max value of 3 color channels in each pixel and copies the result into all channels of that pixel.
 * This is exactly how the Value in HSV image conversion from RGB is calculated. The function takes the source image and caluculates Value for each pixel 
 * and stores the result in all channels of that pixel in destination image.
 *  
 * @param src --> source Image of cv::Mat datatype.
 * @param dst --> destination Image of cv::Mat datatypes.
 * @return 1 if conversion succesful.
 */

int greyscale( cv::Mat &src, cv::Mat &dst)
{
  // types:
  // CV_8U  (uchar greyscale)
  // CV_8UC3  (uchar 3-color image)
  // CV_16S (short)
  // CV_16SC3  (short 3-color imagee)
  // CV_32F  (float, 4-byte per color value)
  // CV_32FC3

   dst = cv::Mat::zeros(src.size(),CV_8UC3);

   for (int i = 0; i < src.rows; i++){
        cv::Vec3b *src_row_ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *dst_row_ptr = dst.ptr<cv::Vec3b>(i);
        for(int j = 0; j < src.cols; j++){
            uint8_t grey_value =  std::max({src_row_ptr[j][0],src_row_ptr[j][1],src_row_ptr[j][2]});
            // uint8_t grey_value =  src_row_ptr[j][2];
            dst_row_ptr[j][0] = grey_value;
            dst_row_ptr[j][1] = grey_value;
            dst_row_ptr[j][2] = grey_value;
        }
   } 
   
   return 1;
}

/**
 * Function to blur a BGR color image using a gaussian 1D kernel of size 5.
 * In this function the associative property of convolution is used to compute the output image.
 * (x1 * x2) * x3 = x1 * (x2 * x3). Instead of applying a single 5x5 kernel to the input image we use two 1D kernels of size 1x5 and 5X1
 * one after the other. The edge and corener cases are handled by only multiplying those values of kernel for which the corresponding pixels values
 * are available. This can also be interpreted as doing zero padding to the source image and then applying the kernel.
 * 
 * @param src --> source Image of cv::Mat datatype.
 * @param dst --> destination Image of cv::Mat datatype.
 * @return 1 if conversion succesful.
 */

int blur5x5( cv::Mat &src, cv::Mat &dst ){

    std::vector<int> blur5x5_Kernel{1,2,4,2,1};
    int kernel_sum = std::accumulate(blur5x5_Kernel.begin(), blur5x5_Kernel.end(),0);

    uint8_t padding = (sizeof(blur5x5_Kernel)/(sizeof(blur5x5_Kernel[0])*2));
    cv::Mat temp  = cv::Mat::zeros(src.size(),CV_8UC3);  
    dst = cv::Mat::zeros(src.size(),CV_8UC3);
    
    for (int i = 0; i < (src.rows); i++){
        cv::Vec3b *src_row_ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *temp_row_ptr = temp.ptr<cv::Vec3b>(i);

        for(int j = padding; j < (src.cols); j++){
            for(uint8_t channel=0;channel<3;channel++) {

                if(j==0){
                    temp_row_ptr[j][channel]  = (src_row_ptr[j][channel]*blur5x5_Kernel[2] + src_row_ptr[j+1][channel]*blur5x5_Kernel[3]+
                                                src_row_ptr[j+2][channel]*blur5x5_Kernel[4])/kernel_sum;

                } else if(j==1){
                    temp_row_ptr[j][channel]  = (src_row_ptr[j-1][channel]*blur5x5_Kernel[1] + src_row_ptr[j][channel]*blur5x5_Kernel[2] + 
                                                src_row_ptr[j+1][channel]*blur5x5_Kernel[3] + src_row_ptr[j+2][channel]*blur5x5_Kernel[4])/kernel_sum;

                } else if(j== (src.cols-2)) {
                    temp_row_ptr[j][channel]  = (src_row_ptr[j-2][channel]*blur5x5_Kernel[0] + src_row_ptr[j-1][channel]*blur5x5_Kernel[1] + 
                                                src_row_ptr[j][channel]*blur5x5_Kernel[2] + src_row_ptr[j+1][channel]*blur5x5_Kernel[3])/kernel_sum;
                } else if(j== (src.cols-1)){
                    temp_row_ptr[j][channel]  = (src_row_ptr[j-2][channel]*blur5x5_Kernel[0] + src_row_ptr[j-1][channel]*blur5x5_Kernel[1] + 
                                                src_row_ptr[j][channel]*blur5x5_Kernel[2])/kernel_sum;
                }
                else{
                    temp_row_ptr[j][channel]   = (src_row_ptr[j][channel]*blur5x5_Kernel[2] + src_row_ptr[j-1][channel]*blur5x5_Kernel[1] + 
                                                src_row_ptr[j-2][channel]*blur5x5_Kernel[0] + src_row_ptr[j+1][channel]*blur5x5_Kernel[3]+
                                                src_row_ptr[j+2][channel]*blur5x5_Kernel[4])/kernel_sum;
                }
            }
        }

    }
    
    for (int i = 0; i < (temp.rows); i++){
        cv::Vec3b *dst_row_ptr = dst.ptr<cv::Vec3b>(i);
        
        cv::Vec3b *temp_row_ptr = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *temp_row_ptr_m1 = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *temp_row_ptr_m2 = src.ptr<cv::Vec3b>(i-2);

        cv::Vec3b *temp_row_ptr_p1 = src.ptr<cv::Vec3b>(i+1);
        cv::Vec3b *temp_row_ptr_p2 = src.ptr<cv::Vec3b>(i+2);
    
        for(int j = 0; j < (src.cols); j++){
            for(uint8_t channel=0;channel<3;channel++) {

                if(i==0){
                    dst_row_ptr[j][channel]  = (temp_row_ptr[j][channel]*blur5x5_Kernel[2] + temp_row_ptr_p1[j][channel]*blur5x5_Kernel[3]+
                                                temp_row_ptr_p2[j][channel]*blur5x5_Kernel[4])/kernel_sum;

                } else if(i==1){
                    dst_row_ptr[j][channel]  = (temp_row_ptr_m1[j][channel]*blur5x5_Kernel[1] + temp_row_ptr[j][channel]*blur5x5_Kernel[2] + 
                                                temp_row_ptr_p1[j][channel]*blur5x5_Kernel[3] + temp_row_ptr_p2[j][channel]*blur5x5_Kernel[4])/kernel_sum;

                } else if(i== (src.rows-2)) {
                    dst_row_ptr[j][channel]  = (temp_row_ptr_m2[j][channel]*blur5x5_Kernel[0] + temp_row_ptr_m1[j][channel]*blur5x5_Kernel[1] + 
                                                temp_row_ptr[j][channel]*blur5x5_Kernel[2] + temp_row_ptr_p1[j][channel]*blur5x5_Kernel[3])/kernel_sum;
                } else if(i== (src.rows-1)){
                    dst_row_ptr[j][channel]  = (temp_row_ptr_m2[j][channel]*blur5x5_Kernel[0] + temp_row_ptr_m1[j][channel]*blur5x5_Kernel[1] + 
                                                temp_row_ptr[j][channel]*blur5x5_Kernel[2])/kernel_sum;
                }
                else{
                    dst_row_ptr[j][channel]   = (temp_row_ptr_m2[j][channel]*blur5x5_Kernel[2] + temp_row_ptr_m1[j][channel]*blur5x5_Kernel[1] + 
                                                temp_row_ptr[j][channel]*blur5x5_Kernel[0] + temp_row_ptr_p1[j][channel]*blur5x5_Kernel[3]+
                                                temp_row_ptr_p2[j][channel]*blur5x5_Kernel[4])/kernel_sum;
                }
            }
        }
    }
    
        
    return 1;
}

/**
 * Function to detect edges ina BGR color image using a sobel 3x3 kernel.
 * Here the kernal applied is specifically for the positive X-> direction. 
 * The edge and corener cases are handled by only multiplying those values of kernel for which the corresponding pixels values
 * are available. This can also be interpreted as doing zero padding to the source image and then applying the kernel.
 * 
 * @param src --> source Image of cv::Mat datatype.
 * @param dst --> destination Image of cv::Mat datatype.
 * @return 1 if conversion succesful.
 */

int sobelX3x3( cv::Mat &src, cv::Mat &dst ){

    int sobel_kernel_x[3][3] = {-1, 0 ,1 ,-2 ,0 , 2 , -1 , 0 , 1};
    int kernel_sum = 4;
    uint8_t padding = (sizeof(sobel_kernel_x[0])/(sizeof(sobel_kernel_x[0][0])*2));
    
    dst = cv::Mat::zeros(src.size(),CV_16SC3);

    for (int i = 0; i < (src.rows); i++){


        cv::Vec3b *src_row_ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *src_row_ptr_m1 = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *src_row_ptr_p1 = src.ptr<cv::Vec3b>(i+1);

        cv::Vec3s *dst_row_ptr = dst.ptr<cv::Vec3s>(i);

        for(int j = padding; j < (src.cols); j++){
            for(uint8_t channel=0;channel<3;channel++) {

                if(i%src.rows ==0){
                    
                    if(j%src.cols == 0){
                        dst_row_ptr[j][channel]   = (src_row_ptr[j+1][channel]*sobel_kernel_x[1][2] +
                                                     src_row_ptr_p1[j+1][channel]*sobel_kernel_x[2][2] )/kernel_sum;
                    }

                    else if(j == (src.cols -1)){
                        dst_row_ptr[j][channel]   = (src_row_ptr[j-1][channel]*sobel_kernel_x[1][0] +
                                                     src_row_ptr_p1[j-1][channel]*sobel_kernel_x[2][0]  )/kernel_sum;
                    }

                    else{
                        dst_row_ptr[j][channel]   = ( src_row_ptr[j-1][channel]*sobel_kernel_x[1][0]     + src_row_ptr[j+1][channel]*sobel_kernel_x[1][2] +
                                            src_row_ptr_p1[j-1][channel]*sobel_kernel_x[2][0]  + src_row_ptr_p1[j+1][channel]*sobel_kernel_x[2][2] )/kernel_sum;
                    }
            
                }
                else if(i == (src.rows -1)){
                    if(j%src.cols == 0){
                        dst_row_ptr[j][channel]   = (src_row_ptr_m1[j+1][channel]*sobel_kernel_x[0][2] + 
                                                     src_row_ptr[j+1][channel]*sobel_kernel_x[1][2])/kernel_sum;
                    }
                    else if(j == (src.cols -1)){
                        dst_row_ptr[j][channel]   = (src_row_ptr_m1[j-1][channel]*sobel_kernel_x[0][0] +
                                                     src_row_ptr[j-1][channel]*sobel_kernel_x[1][0] )/kernel_sum;
                    }
                    else{
                        dst_row_ptr[j][channel]   = (src_row_ptr_m1[j-1][channel]*sobel_kernel_x[0][0] + src_row_ptr_m1[j+1][channel]*sobel_kernel_x[0][2] + 
                                            src_row_ptr[j-1][channel]*sobel_kernel_x[1][0]     + src_row_ptr[j+1][channel]*sobel_kernel_x[1][2])/kernel_sum;
                    }

                }
                else{
                    dst_row_ptr[j][channel]   = (src_row_ptr_m1[j-1][channel]*sobel_kernel_x[0][0] + src_row_ptr_m1[j+1][channel]*sobel_kernel_x[0][2] + 
                                            src_row_ptr[j-1][channel]*sobel_kernel_x[1][0]     + src_row_ptr[j+1][channel]*sobel_kernel_x[1][2] +
                                            src_row_ptr_p1[j-1][channel]*sobel_kernel_x[2][0]  + src_row_ptr_p1[j+1][channel]*sobel_kernel_x[2][2] )/kernel_sum;
                }
            }
        }
    }
    cv::convertScaleAbs(dst,dst,2);
    return 1;
}

/**
 * Function to detect edges in a BGR color image along the positive Y direction using a sobel 3x3 kernel.
 * Here the kernal applied is specifically for the positive Y-> direction. 
 * The edge and corener cases are handled by only multiplying those values of kernel for which the corresponding pixels values
 * are available. This can also be interpreted as doing zero padding to the source image and then applying the kernel.
 * 
 * @param src --> source Image of cv::Mat datatype.
 * @param dst --> destination Image of cv::Mat datatype.
 * @return 1 if conversion succesful.
 */

int sobelY3x3( cv::Mat &src, cv::Mat &dst ){

    int sobel_kernel_x[3][3] = {1, 2 ,1 ,0 ,0 , 0 , -1 , -2 , -1};
    int kernel_sum = 4;
    uint8_t padding = (sizeof(sobel_kernel_x[0])/(sizeof(sobel_kernel_x[0][0])*2));
    
    dst = cv::Mat::zeros(src.size(),CV_16SC3);

    for (int i = 0; i < (src.rows); i++){
        cv::Vec3b *src_row_ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *src_row_ptr_m1 = src.ptr<cv::Vec3b>(i-1);
        cv::Vec3b *src_row_ptr_p1 = src.ptr<cv::Vec3b>(i+1);

        cv::Vec3s *dst_row_ptr = dst.ptr<cv::Vec3s>(i);

        for(int j = padding; j < (src.cols); j++){
            for(uint8_t channel=0;channel<3;channel++) {

                if(i%src.rows ==0){
                    
                    if(j%src.cols == 0){
                        dst_row_ptr[j][channel]   = (src_row_ptr_p1[j][channel]*sobel_kernel_x[2][1] + 
                                                     src_row_ptr_p1[j+1][channel]*sobel_kernel_x[2][2])/kernel_sum;
                    }

                    else if(j == (src.cols -1)){
                        dst_row_ptr[j][channel]   = (src_row_ptr_p1[j-1][channel]*sobel_kernel_x[2][0]  + 
                                                     src_row_ptr_p1[j][channel]*sobel_kernel_x[2][1])/kernel_sum;
                    }

                    else{
                        dst_row_ptr[j][channel]   = (src_row_ptr_p1[j-1][channel]*sobel_kernel_x[2][0]  + 
                                                    src_row_ptr_p1[j][channel]*sobel_kernel_x[2][1] + 
                                                    src_row_ptr_p1[j+1][channel]*sobel_kernel_x[2][2])/kernel_sum;
                    }
            
                }
                else if(i == (src.rows -1)){
                    if(j%src.cols == 0){
                        dst_row_ptr[j][channel]   = (src_row_ptr_m1[j][channel]*sobel_kernel_x[0][1] +
                                                      src_row_ptr_m1[j+1][channel]*sobel_kernel_x[0][2])/kernel_sum;
                        
                    }
                    else if(j == (src.cols -1)){
                        dst_row_ptr[j][channel]   = (src_row_ptr_m1[j-1][channel]*sobel_kernel_x[0][0] + 
                                                    src_row_ptr_m1[j][channel]*sobel_kernel_x[0][1])/kernel_sum;
                    }
                    else{
                        dst_row_ptr[j][channel]   = (src_row_ptr_m1[j-1][channel]*sobel_kernel_x[0][0] + 
                                                    src_row_ptr_m1[j][channel]*sobel_kernel_x[0][1]+
                                                    src_row_ptr_m1[j+1][channel]*sobel_kernel_x[0][2])/kernel_sum;
                    }

                }
                else{
                    dst_row_ptr[j][channel]   = (src_row_ptr_m1[j-1][channel]*sobel_kernel_x[0][0] + src_row_ptr_m1[j][channel]*sobel_kernel_x[0][1] +
                                                src_row_ptr_m1[j+1][channel]*sobel_kernel_x[0][2] + src_row_ptr_p1[j-1][channel]*sobel_kernel_x[2][0]  + 
                                                src_row_ptr_p1[j][channel]*sobel_kernel_x[2][1] + src_row_ptr_p1[j+1][channel]*sobel_kernel_x[2][2])/kernel_sum;
                }
            }
        }
    }
    cv::convertScaleAbs(dst,dst,2);
    return 1;
}

/**
 * Function to convert a BGR color image into magnitude image containing magnitude of the sobel filter.
 * The magnitude of the Sobel filter is a measure of the strength of the edge at a given pixel. 
 * It is calculated by taking the square root of the sum of the squares of the Sobel filter's x and y derivatives.
 * To calculate the magnitude of the Sobel filter in OpenCV, you can first apply the Sobel filter 
 * to the image to obtain the x and y derivatives, and then calculate the magnitude using this magnitude() function.
 * 
 * @param sx --> source Image of cv::Mat datatype obtained after appyling sobel_x filter.
 * @param sy --> source Image of cv::Mat datatype obtained after appyling sobel_y filter.
 * @param dst --> destination Image of cv::Mat datatype format to store magnitude image.
 * 
 * @return 1 if conversion succesful.
 */
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &mag, cv::Mat &theta){

    mag = cv::Mat::zeros(sx.size(),CV_8UC3);
    theta = cv::Mat::zeros(sx.size(),CV_8UC3);
    
    for(int i = 0; i < sx.rows; i++){

        cv::Vec3b *sx_row_ptr = sx.ptr<cv::Vec3b>(i);
        cv::Vec3b *sy_row_ptr = sy.ptr<cv::Vec3b>(i);

        cv::Vec3b *mag_row_ptr = mag.ptr<cv::Vec3b>(i);
        cv::Vec3b *theta_row_ptr = theta.ptr<cv::Vec3b>(i);

        for (int j = 0; j < sx.cols; j++){

            for(uint8_t channel=0;channel<3;channel++){
                mag_row_ptr[j][channel] = sqrt((sx_row_ptr[j][channel]*sx_row_ptr[j][channel]) + (sy_row_ptr[j][channel]*sy_row_ptr[j][channel]));

                double angle = atan2(sy_row_ptr[j][channel], sx_row_ptr[j][channel]);
                int angle_in_degrees = int(angle * (180.0 / M_PI));

                theta_row_ptr[j][channel] = angle_in_degrees;
            }
        }
    }

    return 1;
}


/**
 * Function to convert a BGR color image into magnitude image containing magnitude of the sobel filter.
 * The magnitude of the Sobel filter is a measure of the strength of the edge at a given pixel. 
 * It is calculated by taking the square root of the sum of the squares of the Sobel filter's x and y derivatives.
 * To calculate the magnitude of the Sobel filter in OpenCV, you can first apply the Sobel filter 
 * to the image to obtain the x and y derivatives, and then calculate the magnitude using this magnitude() function.
 * 
 * @param sx --> source grayscale Image of cv::Mat datatype obtained after appyling sobel_x filter.
 * @param sy --> source grayscale Image of cv::Mat datatype obtained after appyling sobel_y filter.
 * @param mag --> destination Image of cv::Mat datatype format to store magnitude image.
 * @param theta --> destination Image of cv::Mat datatype to store the angle
 * @return 1 if conversion succesful.
 */
int magnitude_grayscale( cv::Mat &sx, cv::Mat &sy, cv::Mat &grad){

    grad = cv::Mat::zeros(sx.size(),CV_8UC2);
    
    for(int i = 0; i < sx.rows; i++){

        uchar *sx_row_ptr = sx.ptr<uchar>(i);
        uchar *sy_row_ptr = sy.ptr<uchar>(i);

        cv::Vec2b *dst_row_ptr = grad.ptr<cv::Vec2b>(i);

        for (int j = 0; j < sx.cols; j++){

            dst_row_ptr[j][0] = sqrt((sx_row_ptr[j]*sx_row_ptr[j]) + (sy_row_ptr[j]*sy_row_ptr[j]));

            double angle = atan2(sy_row_ptr[j], sx_row_ptr[j]);
            int angle_in_degrees = int(angle * (180.0 / M_PI));
            dst_row_ptr[j][1] = angle_in_degrees;
        }
    }

    return 1;
}

/**
 * Function to apply blur and quantize effect to a BGR color image.
 * This function takes in the source image and blurs it using the blur5x5 function to smooth the image and remove sharp edges.
 * After blur operation we apply a quantization effect to the image. Quantization is a technique used to reduce the number of colors in an image
 * 
 * @param src --> source Image of cv::Mat datatype.
 * @param dst --> destination Image of cv::Mat datatype.
 * @param levels --> parameter to define the number of colors we want in the output image.
 *                       The final image will have (levels)**3 color values.
 * 
 * @return 1 if conversion succesful.
 */

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){

    int bucket_size = 255/levels;
    // xt = x/b
    // xf = xt*b
    dst = cv::Mat::zeros(src.size(),CV_8UC3);
    
    blur5x5(src,dst);

    for(int i = 0; i < dst.rows; i++){
        cv::Vec3b *dst_row_ptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < dst.cols; j++){

            for(uint8_t channel=0;channel<3;channel++){

                dst_row_ptr[j][channel] = ((dst_row_ptr[j][channel]/bucket_size) * bucket_size);
            }
        }
    }
    return 1;
}

/**
 * Function to apply cartoon effect to a BGR color image.
 * This function takes in the source image and blurs it using the blur5x5 function to smooth the image and remove sharp edges.
 * After blur operation we apply a quantization effect to the image. Quantization is a technique used to reduce the number of colors in an image.
 * The magnitude of the source image is determined paralley by the magnitude function which gives the strength of the edge at any given pixel.
 * We then use the magnitude value to modify the pixels of the blur_quantized image which produces the output image.
 * 
 * @param src --> source Image of cv::Mat datatype.
 * @param dst --> destination Image of cv::Mat datatype.
 * @param levels --> parameter to define the number of colors we want in the output image.
 *                       The final image will have (levels)**3 color values.
 * @param magThreshold -->The level above which all pixels in the blur_quantized image which have a corresponding magnitude value greater than
 *                             magThreshold will be made zero.
 * @return 1 if conversion succesful.
 */

int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold ){

    dst = cv::Mat::zeros(src.size(),CV_8UC3);
    cv::Mat sobelX_frame = cv::Mat::zeros(src.size(),CV_8UC3);
    cv::Mat sobelY_frame = cv::Mat::zeros(src.size(),CV_8UC3);
    cv::Mat magnitude_frame = cv::Mat::zeros(src.size(),CV_8UC3);
    cv::Mat blur_frame = cv::Mat::zeros(src.size(),CV_8UC3);
    

    sobelX3x3(src,sobelX_frame);
    sobelY3x3(src,sobelY_frame);
    magnitude(sobelX_frame,sobelY_frame,magnitude_frame);

    blurQuantize(src,blur_frame,levels);

    for(int i = 0; i < dst.rows; i++){
        cv::Vec3b *dst_row_ptr = dst.ptr<cv::Vec3b>(i);
        cv::Vec3b *blur_row_ptr = blur_frame.ptr<cv::Vec3b>(i);
        cv::Vec3b *mag_row_ptr = magnitude_frame.ptr<cv::Vec3b>(i);

        for (int j = 0; j < dst.cols; j++){

            for(uint8_t channel=0;channel<3;channel++){

                if(mag_row_ptr[j][channel] > magThreshold){
                    dst_row_ptr[j][channel] = 0;

                } else{
                    dst_row_ptr[j][channel] = blur_row_ptr[j][channel];
                }
            }
        }
    }
    return 1;
}

/**
 * Function to apply negative effect to a BGR color image.
 * This function takes in the source image and applies (255 - pixel_value) operation at each channel of every pixel.
 * This produces a negative image with white background.
 * 
 * @param arg1 src --> source Image of cv::Mat datatype.
 * @param arg2 dst --> destination Image of cv::Mat datatype.

 * @return 1 if conversion succesful.
 */

int imageNegative( cv::Mat &src, cv::Mat&dst){

    dst = cv::Mat::zeros(src.size(),CV_8UC3);
    // dst = 255 - src;
    //cv::bitwise_not(src,dst);

    for(int i = 0; i < src.rows; i++){
        cv::Vec3b *dst_row_ptr = dst.ptr<cv::Vec3b>(i);
        cv::Vec3b *src_row_ptr = src.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; j++){

            for(uint8_t channel=0;channel<3;channel++){

                    dst_row_ptr[j][channel] = 255 - src_row_ptr[j][channel];
            }                
        }
    }
    return 0;
}

/**
 * Brief description of the function.
 * Function to convert a BGR color image into grayscale
 * 
 * Detailed explanation of the function, including any inputs, outputs, and edge cases.
 * The function max value of 3 color channels in each pixel and copies them into all channels of that pixel.
 * This is exactly how the Value in HSV is calculated.
 *  
 * @param arg1 src --> source Image of cv::Mat datatype format.
 * @param arg2 dst --> destination Image of cv::Mat datatype format .
 * @return 1 if conversion succesful.
 */
int sparkleOnEdges( cv::Mat& src, cv::Mat& dst){
    cv::Sobel(src, dst, CV_32F, 1, 1);
    cv::convertScaleAbs(dst, dst,2);    
    return 0;
}

/**
 * Function to apply median blur effect to a BGR color image.
 * Median blur is a type of image smoothing filter that replaces the pixel value at the center of the kernel 
 * with the median value of the pixels in the kernel. This can be useful for removing salt-and-pepper noise from an image.
 *  
 * @param src  source Image of cv::Mat datatype format.
 * @param dst  destination Image of cv::Mat datatype format. 
 * @param kernelSize the size of the kernel. It should be a positive odd integer value, e.g. 3, 5, 7.
 * @return 1 if conversion succesful else 0.
 */
int medianBlur(cv::Mat &src, cv::Mat &dst, int kernelSize){

    if(kernelSize%2 == 0){
        std::cerr << "ERROR: Please specify a positive odd integer value in kernel size" << std::endl;
        return 0;
    }
    uint8_t padding = (kernelSize/2);
    cv::Mat temp;
    cv::copyMakeBorder(src, temp, padding, padding, padding, padding, cv::BORDER_REPLICATE);

    dst = cv::Mat::zeros(temp.size(),CV_8UC3);

    cv::copyMakeBorder(src, dst, padding, padding, padding, padding, cv::BORDER_REPLICATE);
    
    int max_border_value = (padding+1);
    for (int i = padding; i < temp.rows -max_border_value; i++){

        cv::Vec3b *dst_row_ptr = dst.ptr<cv::Vec3b>(i);
        for (int j = padding; j < temp.cols -max_border_value; j++){
            for (int channel = 0; channel <3;channel++){
                std::vector<int> median_vector(kernelSize*kernelSize, 0);
                int count = 0;
                for (int k = i-padding; k <= i+padding; k++){
                    cv::Vec3b *temp_row_ptr = temp.ptr<cv::Vec3b>(k);
                    for (int l = j-padding; l <= j+padding; l++){
                        median_vector[count] = temp_row_ptr[l][channel];
                        count++;
                    }
                }

                std::sort(median_vector.begin(), median_vector.end());
                //take the middle element
                int middle = median_vector[median_vector.size() / 2];
                dst_row_ptr[j][channel] = middle;      
            }
        }
    }
    cv::Rect roi(padding, padding,src.cols,src.rows);
    dst = dst(roi);
    return 0;
}


