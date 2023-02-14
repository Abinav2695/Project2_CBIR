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
    std::cout << "Enter the task number for training dataset   " << std::endl;
    std::cout << " '1' : Task 1 --> Baseline Feature Set  " << std::endl;
    std::cout << " '2' : Task 2 --> RG Histogram Feature Set  " << std::endl;
    std::cout << " '3' : Task 2 --> RGB Histogram Feature Set " << std::endl;
    std::cout << " '4' : Task 3 --> RGB Multi Histogram Feature Set " << std::endl;
    std::cout << " '5' : Task 4 --> RGB Colour Histogram and Sobel Gradient Texture Feature Set: " << std::endl;
    std::cout << " '6' : Task 5 --> RGB Colour Histogram and Sobel Gradient Texture Feature Set on Custom Dataset  " << std::endl;
    std::cout << " '7' : Extension 1 --> HSV colour Histogram and Spatial Variance Feature Set on Custom Dataset  " << std::endl;
    std::cout << " '8' : Extension 2 --> RGB Colour Histogram and LAW Filter Texture Feature Set on Custom Dataset " << std::endl;
    std::cout << " '9' : To quit program: " << std::endl;

    int choice = 0;
    bool quit = false;
    std::cin >> choice;

    switch(choice){

        case 0: std::cout << "Please enter a valid choice" << std::endl; break;
        case 1: {
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_1);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[0]);
            // sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[10]);
            FeatureCfg config;
            config.databaseFolder = image_database_path;
            config.csvFile = csv_file_path;
            config.mode = kModeBaselineMatching;
            save_feature_set(&config);
            break;
        }

        case 2: {
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_1);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[1]);
            FeatureCfg config;
            config.databaseFolder = image_database_path;
            config.csvFile = csv_file_path;
            config.mode = kModeHistogramMatching;
            config.type = kRGHistogram;
            save_feature_set(&config);
            break;
        }

        case 3: {
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_1);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[2]);
            FeatureCfg config;
            config.databaseFolder = image_database_path;
            config.csvFile = csv_file_path;
            config.mode = kModeHistogramMatching;
            config.type = kRGBHistogram;
            save_feature_set(&config);
            break;
        }

        case 4: {
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_1);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[3]);
            sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[4]);
            FeatureCfg config;
            config.databaseFolder = image_database_path;
            config.csvFile = csv_file_path;
            config.csvFile2 = csv_file_path2;
            config.mode = kModeMultiHistogramMatching;
            config.type = kRGBHistogram;
            save_feature_set(&config);
            break;
        }

        case 5: {
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_1);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[5]);
            sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[6]);
            FeatureCfg config;
            config.databaseFolder = image_database_path;
            config.csvFile = csv_file_path;
            config.csvFile2 = csv_file_path2;
            config.mode = kModeTextureHOGMMatching;
            config.type = kHOGHistogram;
            save_feature_set(&config);
            break;
        }

        case 6: {
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_2);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[7]);
            sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[8]);
            FeatureCfg config;
            config.databaseFolder = image_database_path;
            config.csvFile = csv_file_path;
            config.csvFile2 = csv_file_path2;
            config.mode = kModeTextureHOGMMatching;
            config.type = kHOGHistogram;
            save_feature_set(&config);
            break;
        }

        case 7: {
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_2);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[11]);
            FeatureCfg config;
            config.databaseFolder = image_database_path;
            config.csvFile = csv_file_path;
            config.mode = kModeSpatialVarMatching;
            config.type = kRGBHistogram;
            save_feature_set(&config);
            break;
        }

        case 8: {
            sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER_2);
            sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[9]);
            sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[10]);
            FeatureCfg config;
            config.databaseFolder = image_database_path;
            config.csvFile = csv_file_path;
            config.csvFile2 = csv_file_path2;
            config.mode = kModeLawFeatureMatching;
            config.type = kLAWHistogram;
            save_feature_set(&config);
            break;
        }

        case 9: std::cout << "Exiting..." << std::endl; quit = true; break;
        default: std::cout << "Please enter a valid choice" << std::endl; break;
    }

    // sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER);
    // sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[9]);
    // sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[10]);
    // std::vector<char *> match_images;
    // float weights[2] = {0.35,0.65};

    // FeatureCfg config;
    // config.databaseFolder = image_database_path;
    // config.csvFile = csv_file_path;
    // config.csvFile2 = csv_file_path2;
    // config.mode = kModeLawFeatureMatching;
    // config.type = kLAWHistogram;
    // // config.type = kRGHistogram;
    // // save_feature_set(&config);

    // MatchCfg mconfig;
    // mconfig.N = 4;
    // mconfig.csvFile = csv_file_path;
    // mconfig.csvFile2 = csv_file_path2;
    // mconfig.mode = kModeLawFeatureMatching;
    // mconfig.type = kLAWHistogram;
    // // mconfig.type = kRGHistogram;
    // mconfig.weights = weights;
    // // mconfig.targetImageFile = "/home/exmachina/NEU/SEM-2/PRCV/Assignment/PRCV_Project2_Image_Retrieval/olympus_3/pic.0535.jpg";
    // mconfig.targetImageFile = "/home/exmachina/NEU/SEM-2/PRCV/Assignment/PRCV_Project2_Image_Retrieval/olympus_3/pic.0711.jpg";
    
    // // mconfig.targetImageFile = "/home/exmachina/NEU/SEM-2/PRCV/Assignment/PRCV_Project2_Image_Retrieval/landscape/00000007_(5).jpg";
    // //beach 00000115_(2).jpg
    // //sunset 00000169_(4).jpg
    // match_feature_set(&mconfig, match_images);

    // for (int i = 0; i < mconfig.N; i++){
    //     char image_path[255];
    //     char window_name[255];
    //     strcpy(image_path, image_database_path);
    //     strcat(image_path, match_images[i]);
    //     sprintf(window_name,"Image_window_%d",i);
    //     cv::Mat frame = cv::imread(image_path);
    //     cv::imshow(window_name,frame);
        
    // }
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // std::cout << "Exiting..." << std::endl;
        if(quit){
            break;
        }
    }
   return 0;
}

