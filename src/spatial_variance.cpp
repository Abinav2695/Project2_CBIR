#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include "filter.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
   // Load an image
    Mat image = imread("/home/exmachina/NEU/SEM-2/PRCV/Assignment/PRCV_Project2_Image_Retrieval/landscape/00000017_(7).jpg", IMREAD_COLOR);

    // Convert the image to the HSV color space
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // Define the number of bins for each channel
    int hbins = 30, sbins = 32, vbins = 32;

    // Define the range of each channel
    float hranges[] = { 0, 180 };
    float sranges[] = { 0, 256 };
    float vranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges, vranges };

    // Split the image into separate color channels
    Mat h, s, v;
    vector<Mat> hsv_channels;
    split(hsv, hsv_channels);
    h = hsv_channels[0];
    s = hsv_channels[1];
    v = hsv_channels[2];

    // Compute the spatial variance of each color bin in the H, S, and V channels
    Mat hist;
    float mean[3] = {0}, stddev[3] = {0};
    for (int i = 0; i < 3; i++) {
        Mat channel = (i == 0) ? h : (i == 1) ? s : v;
        int channels[] = { 0 };
        int dims = 1;

        // Compute the histogram of the channel

        //&channel: The source array(s)
        //1: The number of source arrays (in this case we are using 1. We can enter here also a list of arrays )
        //0: The channel (dim) to be measured. In this case it is just the intensity (each array is single-channel) so we just write 0.
        //Mat(): A mask to be used on the source array ( zeros indicating pixels to be ignored ). If not defined it is not used
        //hist: The Mat object where the histogram will be stored
        //dims: The histogram dimensionality.
        //&hbins: The number of bins per each used dimension
        //ranges: The range of values to be measured per each dimension
        //uniform=true and accumulate=false: The bin sizes are the same and the histogram is cleared at the beginning.

        calcHist(&channel, 1, channels, Mat(), hist, dims, &hbins, ranges, true, false);

        // Normalize the histogram
        // normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
        float pixel_count = channel.rows * channel.cols;
        // Compute the mean and standard deviation of the histogram
        
        
        for (int j = 0; j < hist.rows; j++) {
            float temp = hist.at<float>(j, 0)/pixel_count;
            mean[i] += temp * j;
            stddev[i] += temp * pow(j - mean[i], 2);
        }
        stddev[i] = sqrt(stddev[i] / hist.rows);


        cout << "Channel " << i << ": mean = " << mean[i] << ", stddev = " << stddev[i] << endl;
    }

    
    //////////////////////////////////////////////////
    int h_bins = 8;
    int s_bins = 8;
    int v_bins = 8;

    // Set the number of bins for each channel in an array
    int histSize[] = { h_bins, s_bins, v_bins };

    // Set the ranges for the color channels
    // float h_ranges[] = { 0, 180 };
    // float s_ranges[] = { 0, 256 };
    // float v_ranges[] = { 0, 256 };

    // const float* ranges[] = { h_ranges, s_ranges, v_ranges };

    // Compute the histogram
    cv::Mat hist_op(8, 8, 8, CV_32F);
    int channels[] = { 0, 1, 2 };
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist_op, 3, histSize, ranges, true, false);

    int pixel_count = hsv.rows * hsv.cols;
    // Normalize the histogram
    // cv::normalize(hist_op, hist_op, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
    
    cout << hist_op.dims;

    vector<float> fv;
    int count = 0;
    for (int x = 0; x < hist_op.size[0]; x++)
    {
        for (int y = 0; y < hist_op.size[1]; y++)
        {
            for (int z = 0; z < hist_op.size[2]; z++)
            {
                //count++;
                fv.push_back(hist_op.at<float>(x, y, z)/pixel_count);
                
                //cout << "data " << count << " :" << hist_op.at<float>(x, y, z) << endl;
            }
        }
    }

    return 0;
}



















/*
int main()
{
    // Load the images
    Mat image1 = imread("/home/exmachina/NEU/SEM-2/PRCV/Assignment/PRCV_Project2_Image_Retrieval/kela1.png");
    Mat image2 = imread("/home/exmachina/NEU/SEM-2/PRCV/Assignment/PRCV_Project2_Image_Retrieval/flower.jpg");
    cv::resize(image1,image1,cv::Size(64,128));
    cv::resize(image2,image2,cv::Size(64, 128));

    cv::imshow("image" , image2);
    cv::waitKey(0);
    cout<< image2.size()<< image1.size() << endl;
    // Create a HOG descriptor
    HOGDescriptor hog;

    // Compute the HOG features for the first image
    vector<float> hogFeatures1;
    vector<Point> locations1;
    // hog.compute(image1, hogFeatures1, Size(8, 8), Size(0, 0), locations1);
    hog.compute(image1, hogFeatures1);

    // Compute the HOG features for the second image
    vector<float> hogFeatures2;
    vector<Point> locations2;
    // hog.compute(image2, hogFeatures2, Size(8, 8), Size(0, 0), locations2);
    hog.compute(image2, hogFeatures2);

    // Match the features using Euclidean distance
    cout << hogFeatures1.size() << " " <<hogFeatures2.size() << std::endl;
    vector<DMatch> matches;
    for (int i = 0; i < hogFeatures1.size(); i++) {
        float minDist = FLT_MAX;
        int minIndex = -1;
        for (int j = 0; j < hogFeatures2.size(); j++) {
            // float dist = norm(hogFeatures1[i], hogFeatures2[j]);
            float dist = sqrt( hogFeatures1[i]* hogFeatures1[i] -  hogFeatures2[j]* hogFeatures2[j]);
            if (dist < minDist) {
                minDist = dist;
                minIndex = j;
            }
        }
        if (minDist < 0.5) {
            matches.push_back(DMatch(i, minIndex, minDist));
        }
    }

    // Output the result
    cout << "Number of matches: " << matches.size() << endl;

    return 0;
}
*/
