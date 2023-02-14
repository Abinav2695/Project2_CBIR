/* Abinav Anantharaman
   CS 5330 Spring 2023
   Real time video filtering code
   Source file
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <ctime>
#include "csv_util.h"
#include "feature.h"
#include "filter.h"

char PROJECT_FOLDER[] = "PRCV_Project2_Image_Retrieval/";
char IMAGE_DATABASE_FOLDER_1[] = "olympus/";
char IMAGE_DATABASE_FOLDER_2[] = "olympus_3/";
std::vector<char * > CSV_FILE_NAMES = {"csv_files/baseline_feature.csv",
                                       "csv_files/task2_feature_RG.csv",
                                       "csv_files/task2_feature_RGB.csv",
                                       "csv_files/task3_feature1.csv",
                                       "csv_files/task3_feature2.csv",
                                       "csv_files/task4_color_feature.csv",
                                       "csv_files/task4_texture_feature.csv",
                                       "csv_files/task5_face_color_feature.csv",
                                       "csv_files/task5_face_texture_feature.csv",
                                       "csv_files/extension2_colour_feature.csv",
                                       "csv_files/extension2_law_filter_texture_feature.csv",
                                       "csv_files/extension1_spatial_var_feature.csv",
                                       "csv_files/spatial_var_feature_2.csv",
                                       };

std::string file_path = __FILE__;
std::string PROJECT_FOLDER_PATH = file_path.substr(0, file_path.rfind(PROJECT_FOLDER));

std::vector<char *> images;
std::vector<std::vector<float> > feature_data;

char image_database_path[255];
char csv_file_path[255];
char csv_file_path2[255];


int main(int argc, char* argv[]){

   for(;;){
    std::cout << "Usage :  " ;
    std::cout << "Enter the task number for matching method with imagepath   " << std::endl;
    std::cout << "Example: 1 full_image_path/image.jpg   " << std::endl;
    std::cout << std::endl;
    std::cout << " '1' {image_path} : Task 1 --> Baseline Feature Matching  " << std::endl;
    std::cout << " '2' {image_path}: Task 2 --> RG Histogram Feature Set  " << std::endl;
    std::cout << " '3' {image_path}: Task 2 --> RGB Histogram Feature Set " << std::endl;
    std::cout << " '4' {image_path}: Task 3 --> RGB Multi Histogram Feature Set " << std::endl;
    std::cout << " '5' {image_path}: Task 4 --> RGB Colour Histogram and Sobel Gradient Texture Feature Set: " << std::endl;
    std::cout << " '6' {image_path}: Task 5 --> RGB Colour Histogram and Sobel Gradient Texture Feature Set on Custom Dataset  " << std::endl;
    std::cout << " '7' {image_path}: Extension 1 --> HSV colour Histogram and Spatial Variance Feature Set on Custom Dataset  " << std::endl;
    std::cout << " '8' {image_path}: Extension 2 --> RGB Colour Histogram and LAW Filter Texture Feature Set on Custom Dataset " << std::endl;
    std::cout << " '9' : To quit program: " << std::endl;

    std::string input;
    std::getline(std::cin, input); // takes input with spaces
    std::istringstream iss(input);

    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }

    bool quit = false;
    std::vector<char *> match_images;
    MatchCfg mconfig;
    std::cout << tokens[0][0] << std::endl;
    char choice  = tokens[0][0];

    if(tokens.size() == 1){
        std::cout << "Please provide image path" << std::endl;
        exit(1);
    }
    std::cout << "aadsa" << std::endl;
    
    switch(choice){

        case '0': std::cout << "Please enter a valid choice" << std::endl; break;
        case '1': {
            
            char target_image_path[255];
            for (int i = 0; i < tokens[1].size(); i++){
                target_image_path[i] = tokens[1][i];
            }
            target_image_path[tokens[1].size()] = '\0';

            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_1);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[0]);
            // sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[10]);
            // float weights[2] = {0.35,0.65};
            mconfig.N = 4;
            mconfig.csvFile = csv_file_path;
            // mconfig.csvFile2 = csv_file_path2;
            mconfig.mode = kModeBaselineMatching;
            // mconfig.type = kLAWHistogram;
            // mconfig.weights = weights;
            mconfig.targetImageFile = target_image_path;
            match_feature_set(&mconfig, match_images);
            break;
        }

        case '2': {
            char target_image_path[255];
            for (int i = 0; i < tokens[1].size(); i++){
                target_image_path[i] = tokens[1][i];
            }
            target_image_path[tokens[1].size()] = '\0';
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_1);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[1]);
            // sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[10]);
            // float weights[2] = {0.35,0.65};
            mconfig.N = 4;
            mconfig.csvFile = csv_file_path;
            // mconfig.csvFile2 = csv_file_path2;
            mconfig.mode = kModeHistogramMatching;
            mconfig.type = kRGHistogram;
            // mconfig.weights = weights;
            mconfig.targetImageFile = target_image_path;
            match_feature_set(&mconfig, match_images);
            break;
            
        }

        case '3': {
        char target_image_path[255];
            for (int i = 0; i < tokens[1].size(); i++){
                target_image_path[i] = tokens[1][i];
            }
            target_image_path[tokens[1].size()] = '\0';
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_1);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[2]);
            // sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[10]);
            // float weights[2] = {0.35,0.65};
            mconfig.N = 4;
            mconfig.csvFile = csv_file_path;
            // mconfig.csvFile2 = csv_file_path2;
            mconfig.mode = kModeHistogramMatching;
            mconfig.type = kRGBHistogram;
            // mconfig.weights = weights;
            mconfig.targetImageFile = target_image_path;
            match_feature_set(&mconfig, match_images);
            break;
        }

        case '4': {
            char target_image_path[255];
            for (int i = 0; i < tokens[1].size(); i++){
                target_image_path[i] = tokens[1][i];
            }
            target_image_path[tokens[1].size()] = '\0';
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_1);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[3]);
            sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[4]);
            float weights[2] = {0.4,0.6};
            mconfig.N = 4;
            mconfig.csvFile = csv_file_path;
            mconfig.csvFile2 = csv_file_path2;
            mconfig.mode = kModeMultiHistogramMatching;
            mconfig.type = kRGBHistogram;
            mconfig.weights = weights;
            mconfig.targetImageFile = target_image_path;
            match_feature_set(&mconfig, match_images);
            break;
        }

        case '5': {
            char target_image_path[255];
            for (int i = 0; i < tokens[1].size(); i++){
                target_image_path[i] = tokens[1][i];
            }
            target_image_path[tokens[1].size()] = '\0';
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_1);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[5]);
            sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[6]);
            float weights[2] = {0.5,0.5};
            mconfig.N = 4;
            mconfig.csvFile = csv_file_path;
            mconfig.csvFile2 = csv_file_path2;
            mconfig.mode = kModeTextureHOGMMatching;
            mconfig.type = kHOGHistogram;
            mconfig.weights = weights;
            mconfig.targetImageFile = target_image_path;
            match_feature_set(&mconfig, match_images);
            break;
        }

        case '6': {
            char target_image_path[255];
            for (int i = 0; i < tokens[1].size(); i++){
                target_image_path[i] = tokens[1][i];
            }
            target_image_path[tokens[1].size()] = '\0';
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_2);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[7]);
            sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[8]);
            float weights[2] = {0.3,0.7};
            mconfig.N = 4;
            mconfig.csvFile = csv_file_path;
            mconfig.csvFile2 = csv_file_path2;
            mconfig.mode = kModeTextureHOGMMatching;
            mconfig.type = kHOGHistogram;
            mconfig.weights = weights;
        mconfig.targetImageFile = target_image_path;
            match_feature_set(&mconfig, match_images);
            break;
        }

        case '7': {
            char target_image_path[255];
            for (int i = 0; i < tokens[1].size(); i++){
                target_image_path[i] = tokens[1][i];
            }
            target_image_path[tokens[1].size()] = '\0';
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_2);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[11]);
            // sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[10]);
            // float weights[2] = {0.35,0.65};
            mconfig.N = 4;
            mconfig.csvFile = csv_file_path;
            // mconfig.csvFile2 = csv_file_path2;
            mconfig.mode = kModeSpatialVarMatching;
            mconfig.type = kRGBHistogram;
            // mconfig.weights = weights;
            mconfig.targetImageFile = target_image_path;
            match_feature_set(&mconfig, match_images);
            break;
        }

        case '8': {
            char target_image_path[255];
            for (int i = 0; i < tokens[1].size(); i++){
                target_image_path[i] = tokens[1][i];
            }
            target_image_path[tokens[1].size()] = '\0';
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_2);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[9]);
            sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[10]);
            float weights[2] = {0.35,0.65};
            mconfig.N = 4;
            mconfig.csvFile = csv_file_path;
            mconfig.csvFile2 = csv_file_path2;
            mconfig.mode = kModeLawFeatureMatching;
            mconfig.type = kLAWHistogram;
            mconfig.weights = weights;
            mconfig.targetImageFile = target_image_path;
            match_feature_set(&mconfig, match_images);
            break;
        }

        case '9': std::cout << "Exiting..." << std::endl; quit = true; break;
        default: std::cout << "Please enter a valid choice" << std::endl; break;
    }

    for (int i = 0; i < mconfig.N; i++){
        char image_path[255];
        char window_name[255];
        strcpy(image_path, image_database_path);
        strcat(image_path, match_images[i]);
        sprintf(window_name,"Image_window_%d",i);
        cv::Mat frame = cv::imread(image_path);
        cv::imshow(window_name,frame);
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
    if(quit){
        break;
    }

    std::cout << std::endl;
    std::cout << std::endl;
    }

   return 0;
}

