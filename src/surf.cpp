#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    // Load the images
    Mat image1 = imread("image1.jpg", IMREAD_GRAYSCALE);
    Mat image2 = imread("image2.jpg", IMREAD_GRAYSCALE);

    // Detect SURF features
    Ptr<SURF> surf = SURF::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    surf->detectAndCompute(image1, noArray(), keypoints1, descriptors1);
    surf->detectAndCompute(image2, noArray(), keypoints2, descriptors2);

    // Match the features
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // Draw the matches
    Mat imageMatches;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, imageMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Show the result
    imshow("Matches", imageMatches);
    waitKey();

    return 0;
}

