/* Abinav Anantharaman
   CS 5330 Spring 2023
   Filter library
   Include file
*/

//Function to convert a BGR color image into grayscale
int greyscale( cv::Mat &src, cv::Mat &dst);

//Function to blur a BGR color image using a gaussian 1D kernel of size 5.
int blur5x5( cv::Mat &src, cv::Mat &dst );

//Function to detect edges in a BGR color image along the positive X direction using a sobel 3x3 kernel.
int sobelX3x3( cv::Mat &src, cv::Mat &dst );

//Function to detect edges in a BGR color image along the positive Y direction using a sobel 3x3 kernel.
int sobelY3x3( cv::Mat &src, cv::Mat &dst );

// Function to convert a BGR color image into magnitude image containing magnitude of the sobel filter.
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &mag, cv::Mat &theta );
int magnitude_grayscale( cv::Mat &sx, cv::Mat &sy, cv::Mat &grad);

// Function to apply blur and quantize effect to a BGR color image.
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

// Function to apply cartoon effect to a BGR color image.
int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );

// Function to apply negative effect on a BGR color image
int imageNegative( cv::Mat &src, cv::Mat&dst);

// Function to apply sparkel effect to a BGR color image
int sparkleOnEdges( cv::Mat& src, cv::Mat& dst);

// Function to apply mdeain blur smoothing effect to a BGR color image
int medianBlur(cv::Mat &src, cv::Mat &dst, int kernelSize = 3);