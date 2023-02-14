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
    Mat image = imread("/home/exmachina/NEU/SEM-2/PRCV/Assignment/PRCV_Project2_Image_Retrieval/olympus/pic.0324.jpg", IMREAD_GRAYSCALE);


    // Define law  filter kernels
 
    Mat l5e5RT = (Mat_<float>(5,5) <<        -1   , -3    ,-3  ,  -1 ,    0,
                                            -3,    -8  ,  -6 ,    0,     1,
                                            -3   , -6   ,  0  ,   6   ,  3,
                                              -1  ,   0,     6  ,   8  ,   3,
                                             0,    1  ,   3   ,  3  ,   1);


    Mat l5s5RT = (Mat_<float>(5,5) <<       -1 ,   -2  ,  -2   , -2  ,  -1,
                                            -2   ,  0    , 4     ,0   , -2,
                                            -2  ,   4  ,  12 ,    4  ,  -2,
                                             -2  ,   0   ,  4  ,   0  ,  -2,
                                            -1 ,   -2   , -2 ,   -2,    -1);

    Mat l5w5RT = (Mat_<float>(5,5) <<       -1  ,  -1,    -3  ,  -3   ,  0,
                                            -1  ,   8,    6  ,   0   ,  3,
                                            -3  ,   6  ,   0 ,   -6   ,  3,
                                              -3 ,    0  ,  -6 ,   -8 ,    1,
                                             0 ,    3  ,   3 ,    1 ,    1);

    Mat l5r5RT = (Mat_<float>(5,5) <<        1,     0 ,    6 ,    0  ,   1,
                                            0 ,  -16,     0 ,  -16  ,   0,
                                            6   ,  0  ,  36     ,0   ,  6,
                                             0 ,  -16 ,    0   ,-16   ,  0,
                                           1  ,   0  ,   6  ,   0  ,   1);

    Mat e5s5RT = (Mat_<float>(5,5) <<       1 ,    1 ,   -1 ,   -1  ,   0,
                                            1  ,   0 ,   -2,     0   ,  1,
                                           -1 ,   -2 ,    0  ,   2  ,   1,
                                            -1 ,    0  ,   2 ,    0   , -1,
                                             0,     1  ,   1 ,   -1  ,  -1);

    Mat e5w5RT = (Mat_<float>(5,5) <<        1 ,    0 ,    0,     0 ,   -1,
                                            0  ,  -4  ,   0 ,    4   ,  0,
                                             0  ,   0   ,  0   ,  0  ,   0,
                                           0    , 4  ,   0 ,   -4   ,  0,
                                            -1   ,  0   ,  0  ,   0  ,   1);

    Mat e5r5RT = (Mat_<float>(5,5) <<        -1,     1  ,  -3 ,    3  ,   0,
                                            1  ,   8 ,   -6   ,  0,   -3,
                                             -3  ,  -6 ,    0     ,6   ,  3,
                                              3 ,    0,     6,    -8 ,   -1,
                                             0   , -3 ,   3  ,  -1 ,    1);

    Mat s5w5RT = (Mat_<float>(5,5) <<        1  ,  -1 ,   -1  ,   1  ,   0,
                                           -1 ,    0  ,   2  ,   0    ,-1,
                                           -1 ,    2 ,    0   , -2   ,  1,
                                           1  ,   0 ,   -2  ,   0   ,  1,
                                            0  ,  -1 ,    1 ,    1  ,  -1);

    Mat s5r5RT = (Mat_<float>(5,5) <<       -1 ,    2 ,   -2 ,    2 ,   -1,
                                            2   ,  0   , -4  ,   0   ,  2,
                                           -2  ,  -4   , 12  ,  -4 ,   -2,
                                            2 ,    0  ,  -4    , 0 ,    2,
                                           -1   ,  2    ,-2    , 2  ,  -1);
    
    Mat w5r5RT = (Mat_<float>(5,5) <<        -1   ,  3  ,  -3  ,   1 ,    0,
                                            3 ,   -8   ,  6   ,  0  ,  -1,
                                            -3  ,   6  ,   0    ,-6   ,  3,
                                            1  ,   0 ,   -6 ,    8  ,  -3,
                                             0  ,  -1 ,    3 ,   -3    , 1);


    Mat e5e5RT = (Mat_<float>(5,5) <<        1 ,    2  ,   0 ,   -2,    -1,
                                            2 ,    4 ,    0,    -4  ,  -2,
                                            0  ,   0 ,    0    , 0    , 0,
                                            -2   , -4  ,   0  ,   4  ,   2,
                                            -1 ,   -2  ,   0  ,   2   ,  1);

    Mat s5s5RT = (Mat_<float>(5,5) <<         1 ,    0   , -2 ,    0 ,    1,
                                                0 ,    0  ,   0 ,    0 ,    0,
                                                -2 ,    0 ,    4  ,   0  ,  -2,
                                                0  ,   0  ,  0    , 0    , 0,
                                                1 ,    0 ,   -2 ,    0,     1);

    Mat w5w5RT = (Mat_<float>(5,5) <<         1  ,  -2 ,    0 ,    2 ,   -1,
                                                -2 ,    4 ,    0  ,  -4 ,    2,
                                                0,     0     ,0     ,0  ,   0,
                                                2,    -4   ,  0    , 4    ,-2,
                                                -1  ,   2  ,   0  ,  -2  ,   1);

    Mat r5r5RT = (Mat_<float>(5,5) <<          1   , -4  ,   6 ,   -4 ,    1,
                                                -4 ,   16,   -24  ,  16  ,  -4,
                                                6  , -24  , 36  , -24  ,   6,
                                                -4  ,  16,   -24  ,  16   , -4,
                                                1    ,-4  ,   6  ,  -4   ,  1);

    std::vector<cv::Mat> law_kernels;

    law_kernels.push_back(l5e5RT);
    law_kernels.push_back(l5s5RT);
    law_kernels.push_back(l5w5RT);
    law_kernels.push_back(l5r5RT);
    law_kernels.push_back(e5s5RT);
    law_kernels.push_back(e5w5RT);
    law_kernels.push_back(e5r5RT);
    law_kernels.push_back(s5w5RT);
    law_kernels.push_back(s5r5RT);
    law_kernels.push_back(w5r5RT);
    law_kernels.push_back(e5e5RT);
    law_kernels.push_back(s5s5RT);
    law_kernels.push_back(w5w5RT);
    law_kernels.push_back(r5r5RT);

    int num_kernels = 14;
    std::vector<cv::Mat> filtered_images;
    

    for (int i = 0; i < num_kernels;i++){

        cv::Mat result;
        filter2D(image, result, CV_16SC1, law_kernels[i], Point(-1,-1), 0, BORDER_REPLICATE);
        cv::convertScaleAbs(result,result,1);
        filtered_images.push_back(result);
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        std::cout << minVal << " " << maxVal << std::endl;
        // cv::minMaxLoc(result2, &minVal, &maxVal, &minLoc, &maxLoc);
        // std::cout << minVal << " " << maxVal << std::endl;

        imshow("Result", result);
        waitKey(0);
    }

    std::vector<float> feature_vector;

    int bin_size = 256/16;
    for (int i=0 ;i < num_kernels; i++){
        cv::Mat frame = filtered_images[i];
        // cv::Mat left(fram, cv::Rect(0, 0, s.width / 2, s.height / 2));
        // cv::Mat right(img, cv::Rect(s.width / 2, 0, s.width / 2, s.height / 2));
        int pixel_count = frame.rows * frame.cols;
        int histogram[16] = {0};
        for (int row=0; row<frame.rows; ++row){
            uchar *row_ptr = frame.ptr<uchar>(row);
            for (int col=0; col<frame.cols; ++col){
                int bin_index = row_ptr[col]/bin_size;
                histogram[bin_index] +=1;
            }
        }

        for (int count =0; count<16; ++count){
            feature_vector.push_back(histogram[count]/pixel_count);
        }

    }

    std::cout << "Feature set size :: " << feature_vector.size() << std::endl;

    // Mat l5l5_norm;
    // Mat l5s5_norm;
    // cv::divide(l5l5,256,l5l5_norm);
    // cv::divide(l5s5,32,l5s5_norm);
    // // Create a destination image to store the result
    // Mat result1 ;
    // Mat result2 ;
    // Mat result3  = cv::Mat::zeros(image.size(),CV_32F);
    // // Apply the filter using the filter2D function
    // filter2D(image, result1, CV_16SC1, r5r5RT, Point(-1,-1), 0, BORDER_REPLICATE);
    // filter2D(image, result2, -1, l5l5_norm, Point(-1,-1), 0, BORDER_REPLICATE);
    // cv::convertScaleAbs(result1,result1,1);
    // // cv::convertScaleAbs(result2,result2,2);
    // // cv::divide(result1, result2, result3);
    // // Show the result
    // // Get the datatype of the image
    // // int type = result.type();

    // // for (int i = 0; i < result1.rows; i++){
    // //     cv::int16_t* mat_ptr = result1.ptr<cv::int16_t>(i);
    // //     cv::int16_t* mat_ptr1 = result2.ptr<cv::int16_t>(i);

    // //     float* mat_ptr2 = result3.ptr<float>(i);
    // //     for(int j = 0; j < result1.cols; j++){
    // //         mat_ptr2[j] = static_cast<float> (mat_ptr[j])/ static_cast<float>(mat_ptr1[j]) ;
    // //         if(mat_ptr2[j]>1){
    // //             cout << " More than one" << endl;
    // //         }
    // //     }
    // // }

    // imshow("Result", result2);
    // waitKey(0);
    // imshow("Result", result3);
    // waitKey(0);
    cv::destroyAllWindows();
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
