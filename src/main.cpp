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
char IMAGE_DATABASE_FOLDER[] = "olympus_3/";
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

   sprintf(image_database_path,"%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER,IMAGE_DATABASE_FOLDER);
   sprintf(csv_file_path, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[9]);
   sprintf(csv_file_path2, "%s%s%s",PROJECT_FOLDER_PATH.c_str(),PROJECT_FOLDER, CSV_FILE_NAMES[10]);
   std::vector<char *> match_images;
   float weights[2] = {0.35,0.65};

   FeatureCfg config;
   config.databaseFolder = image_database_path;
   config.csvFile = csv_file_path;
   config.csvFile2 = csv_file_path2;
   config.mode = kModeLawFeatureMatching;
   config.type = kLAWHistogram;
   // config.type = kRGHistogram;
   // save_feature_set(&config);

   MatchCfg mconfig;
   mconfig.N = 4;
   mconfig.csvFile = csv_file_path;
   mconfig.csvFile2 = csv_file_path2;
   mconfig.mode = kModeLawFeatureMatching;
   mconfig.type = kLAWHistogram;
   // mconfig.type = kRGHistogram;
   mconfig.weights = weights;
   // mconfig.targetImageFile = "/home/exmachina/NEU/SEM-2/PRCV/Assignment/PRCV_Project2_Image_Retrieval/olympus_3/pic.0535.jpg";
   mconfig.targetImageFile = "/home/exmachina/NEU/SEM-2/PRCV/Assignment/PRCV_Project2_Image_Retrieval/olympus_3/pic.0711.jpg";
   
   // mconfig.targetImageFile = "/home/exmachina/NEU/SEM-2/PRCV/Assignment/PRCV_Project2_Image_Retrieval/landscape/00000007_(5).jpg";
   //beach 00000115_(2).jpg
   //sunset 00000169_(4).jpg
   match_feature_set(&mconfig, match_images);

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

   std::cout << "Exiting..." << std::endl;
}

