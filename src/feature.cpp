/* Abinav Anantharaman
   CS 5330 Spring 2023
   Real time video filtering code
   Source file
*/

#include <iostream>
#include <string>
#include <ctime>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "feature.h"
#include "csv_util.h"
#include "filter.h"



// Compute feature vectors for all images in a database.
// Compute feature vectors based on the mode of feature extraction.
//  MODE0 -  BASELINE_MATCHING
//  MODE1 -  HISTOGRAM_MATCHING
//  MODE2 -  MULTI_HISTOGRAM_MATCHING
//  MODE3 -  TEXTTURE_COLOR_MATCHING

int save_feature_set(FeatureCfg *cfg){
    
    switch(cfg->mode){
        case kModeBaselineMatching: baseline_feature_extraction(cfg); break;
        case kModeHistogramMatching: single_hist_feature_extraction(cfg); break;
        case kModeMultiHistogramMatching: multi_hist_feature_extraction(cfg); break;
        case kModeTextureHOGMMatching: extract_texture_feature(cfg); break;
        // case kModeLawFeatureMatching: extract_law_feature(cfg); break;
        case kModeLawFeatureMatching: law_texture_feature(cfg); break;
        case kModeSpatialVarMatching: extract_spatial_var_feature(cfg); break;
    }
    return 0;
}

// Compare feature vectors for all images in a database wrt to a target immage.
// Compare feature vectors based on the mode of feature extraction.
int match_feature_set(MatchCfg *cfg, std::vector<char *> &match_images){
    
    switch(cfg->mode){
        case kModeBaselineMatching: compare_baseline_feature(cfg, match_images); break;
        case kModeHistogramMatching: single_hist_compare(cfg, match_images); break;
        case kModeMultiHistogramMatching: multi_hist_compare(cfg, match_images); break;
        case kModeTextureHOGMMatching: texture_color_hist_compare(cfg, match_images); break;
        // case kModeLawFeatureMatching: compare_law_features(cfg, match_images); break;
        case kModeLawFeatureMatching: law_texture_hist_compare(cfg, match_images); break;
        case kModeSpatialVarMatching: compare_spatial_var_features(cfg, match_images); break;
    }
    return 1;
}


int get_baseline_feature(char *image, std::vector<float> &feature_vector){

    // // Using a 9x9 box at the center of the image as feature
    // // This wil produce 9x9x3 features for BGR 3 channel image 
    // // The feature vector length will be 9x9x3 = 273

    cv::Mat frame = cv::imread(image,cv::IMREAD_COLOR);
    int kernel_size = 9;

    int row_min_index = frame.rows/2 - (kernel_size/2);
    int row_max_index = frame.rows/2 + ((kernel_size/2) + 1);

    int col_min_index = frame.cols/2 - (kernel_size/2);
    int col_max_index = frame.cols/2 + ((kernel_size/2) + 1);

    int image_channels  = frame.channels();
    
    for (int i = row_min_index; i < row_max_index; i++){
        cv::Vec3b *row_ptr = frame.ptr<cv::Vec3b>(i);
        for (int j = col_min_index; j < col_max_index; j++){
            for (int k = 0; k < image_channels; k++){
                feature_vector.push_back(row_ptr[j][k]);
            }
        }
    }    
    return 0;
}

int baseline_feature_extraction(FeatureCfg *cfg){

    ImageDirectoryAccess img_dir;
    img_dir.rewrite = 1;
    int loop_count = 0;
     // open the directory
    img_dir.dirp = opendir( cfg->databaseFolder );
    if( img_dir.dirp == NULL) {
        printf("Cannot open directory %s\n", cfg->databaseFolder);
        exit(-1);
    }

    // loop over all the files in the image file listing
    while( (img_dir.dp = readdir(img_dir.dirp)) != NULL ) {
        
    // check if the file is an image
        if( strstr(img_dir.dp->d_name, ".jpg") ||
            strstr(img_dir.dp->d_name, ".jpeg") || 
            strstr(img_dir.dp->d_name, ".png") ||
            strstr(img_dir.dp->d_name, ".ppm") ||
            strstr(img_dir.dp->d_name, ".tif") ) 
            {

                loop_count += 1;
                if (loop_count>1){
                    img_dir.rewrite = 0;
                }    
                printf("processing image file: %s\n", img_dir.dp->d_name);

                // build the overall filename
                strcpy(img_dir.imageFullPath, cfg->databaseFolder);
                strcat(img_dir.imageFullPath, "/");
                strcat(img_dir.imageFullPath, img_dir.dp->d_name);
                strcpy(img_dir.imageFileName, img_dir.dp->d_name);

                //compute feature vector
                std::vector<float>feature_vector;
                get_baseline_feature(img_dir.imageFullPath,feature_vector);

                printf("Appending Image Feature Vector... \n");
                append_image_data_csv(cfg->csvFile,img_dir.imageFileName,feature_vector,img_dir.rewrite);
                //printf("full path name: %s\n", img_dir.buffer);
        }

    }
    return 0;
}

float get_baseline_distance_metric(std::vector<float> &fv1, std::vector<float> &fv2){

    double sum = 0.0;
    for (int i=0; i< fv1.size(); i++){

        sum += (fv1[i] - fv2[i]) * (fv1[i] - fv2[i]);
    }

    return sqrt(sum);
}

int compare_baseline_feature(MatchCfg *cfg, std::vector<char *> &match_images){

    //compute target feature vector
    std::vector<float>target_feature_vector;
    get_baseline_feature(cfg->targetImageFile,target_feature_vector);    

    //read image database and feature vectors from csv file
    std::vector<std::vector<float> > feature_set;
    std::vector<char *> images;
    read_image_data_csv(cfg->csvFile,images,feature_set,0);

    //store dustance metric values for each image wrt target image
    std::map<float, char*> target_map;

    printf("Computing Distance Metric... \n");
    for(int i=0; i< images.size(); i++){

        float bs_metric = get_baseline_distance_metric(target_feature_vector,feature_set[i]);
        target_map[bs_metric] = images[i];
    }

    printf("Top %d matches for Target Image %s ::: \n",cfg->N,cfg->targetImageFile);
    int count = 0;
    for (auto it = target_map.begin(); it != target_map.end() && count < cfg->N; ++it, ++count) {
        std::cout << it->first << ": " << it->second << std::endl;
        match_images.push_back(it->second);
    }
    
    return 0;
}


//his feature2
int get_hist_feature(cv::Mat &image, HistogramType type, std::vector<float> &fv, uint8_t size_to_consider) {

int row_start, row_end, col_start, col_end;

switch (size_to_consider){   //size of image to be considered
    case 1:
    row_start = 0;
    col_start = 0;
    row_end = image.rows;
    col_end = image.cols;
    break;

    case 2:
    row_start = (2 * image.rows)/8;
    col_start = (2 * image.rows)/8;
    row_end = (6 * image.rows)/8;
    col_end = (6 * image.rows)/8;
    break;

    case 3:
    row_start = (3 * image.rows)/8;
    col_start = (3 * image.rows)/8;
    row_end = (5 * image.rows)/8;
    col_end = (5 * image.rows)/8;
    break;
}

int nrows  = row_end - row_start;
int ncols = col_end - col_start;
int pixel_count = nrows * ncols;


    switch (type) {

        case kRGBHistogram: { 
            //3D matrix of size binSize[0]*binSize[1]*binSize[2]
            int nRbins = 8;
            int nGbins = 8;
            int nBbins = 8;

            int b_bin_size = 256/(nBbins);
            int g_bin_size = 256/(nGbins);
            int r_bin_size = 256/(nRbins);

            int histogram[nRbins][nGbins][nBbins]; 
            
            // Operate for fv1 and repeat for fv2
            //initialize values to 0 to avoid garbage value
            for (int i = 0; i < nRbins; i++){
                for (int j = 0; j < nGbins; j++){
                    for (int k = 0; k < nBbins; k++){
                        
                        histogram[i][j][k] = 0;
                    }
                }
            }

            // for fv1 we are using full image data
            for (int row = row_start; row < row_end; ++row){
                
                cv::Vec3b *row_ptr = image.ptr<cv::Vec3b>(row);
                for (int col = col_start; col < col_end; ++col){
                    int r_idx, g_idx, b_idx;

                    // find color channel bin index
                    r_idx = row_ptr[col][2]/ r_bin_size;
                    g_idx = row_ptr[col][1]/ g_bin_size;
                    b_idx = row_ptr[col][0]/ b_bin_size;

                    histogram[r_idx][g_idx][b_idx] += 1; // Increment bin value by 1
                }
            }

            for (int i = 0; i < nRbins; i++){
                for (int j = 0; j < nGbins; j++){
                    for (int k = 0; k < nBbins; k++){
                        fv.push_back((float)(histogram[i][j][k])/(pixel_count));
                    }
                }
            }
            break;
        }


        case kGBHistogram: { 

            int nGbins = 16;
            int nBbins = 16;

            int b_bin_size = 256/(nBbins);
            int g_bin_size = 256/(nGbins);
            
            int histogram[nGbins][nBbins]; 
            // Operate for fv1 and repeat for fv2
            //initialize values to 0 to avoid garbage value
            for (int i = 0; i < nGbins; i++){
                for (int j = 0; j < nBbins; j++){
   
                        histogram[i][j] = 0;
                }
            }

            // for fv1 we are using full image data
            for (int row = row_start; row < row_end; ++row){
                
                cv::Vec3b *row_ptr = image.ptr<cv::Vec3b>(row);
                for (int col = col_start; col < col_end; ++col){
                    int  g_idx, b_idx;

                    g_idx = row_ptr[col][1]/ g_bin_size;
                    b_idx = row_ptr[col][0]/ b_bin_size;

                    histogram[g_idx][b_idx] += 1; // Increment bin value by 1
                }
            }

            for (int i = 0; i < nGbins; i++){
                for (int j = 0; j < nBbins; j++){
                        fv.push_back((float)(histogram[i][j])/(pixel_count));
                    }
                }
            break;
        }

        case kRGHistogram: { 
           
            int nGbins = 16;
            int nRbins = 16;

            int r_bin_size = 256/(nRbins);
            int g_bin_size = 256/(nGbins);
            
            int histogram[nRbins][nGbins]; 
            
            // Operate for fv1 and repeat for fv2
            // initialize values to 0 to avoid garbage value
            for (int i = 0; i < nRbins; i++){
                for (int j = 0; j < nGbins; j++){
   
                    histogram[i][j] = 0;
                }
            }
            // Get histogram count based on each pixel
            for (int row = row_start; row < row_end; ++row){
                
                cv::Vec3b *row_ptr = image.ptr<cv::Vec3b>(row);
                for (int col = col_start; col < col_end; ++col){
                    int  r_idx, g_idx, b_idx;

                    r_idx = row_ptr[col][2]/ r_bin_size;
                    g_idx = row_ptr[col][1]/ g_bin_size;

                    histogram[r_idx][g_idx] += 1; // Increment bin value by 1
                }
            }
            for (int i = 0; i < nRbins; i++){
                for (int j = 0; j < nGbins; j++){
                        fv.push_back((float)(histogram[i][j])/(pixel_count));
                    }
                }
            break;
        }

        case kRBHistogram: { 
            int nBbins = 16;
            int nRbins = 16;

            int r_bin_size = 256/(nRbins);
            int b_bin_size = 256/(nBbins);
            
            int histogram[nRbins][nBbins]; 
            
            // Operate for fv1 and repeat for fv2
            // initialize values to 0 to avoid garbage value
            for (int i = 0; i < nRbins; i++){
                for (int j = 0; j < nBbins; j++){
   
                    histogram[i][j] = 0;
                }
            }
            // Get histogram count based on each pixel
            for (int row = row_start; row < row_end; ++row){
                
                cv::Vec3b *row_ptr = image.ptr<cv::Vec3b>(row);
                for (int col = col_start; col < col_end; ++col){
                    int  r_idx, g_idx, b_idx;

                    r_idx = row_ptr[col][2]/ r_bin_size;
                    b_idx = row_ptr[col][0]/ b_bin_size;

                    histogram[r_idx][b_idx] += 1; // Increment bin value by 1
                }
            }
            for (int i = 0; i < nRbins; i++){
                for (int j = 0; j < nBbins; j++){
                        fv.push_back((float)(histogram[i][j])/(pixel_count));
                    }
                }
            break;
        }

        case kHOGHistogram: { 
           
            int nMagbins = 16;
            int nThetabins = 16;

            int mag_bin_size = 256/(nMagbins);
            int theta_bin_size = 180/(nThetabins);
            
            int histogram[nMagbins][nThetabins]; 
            
            // Operate for fv1 and repeat for fv2
            // initialize values to 0 to avoid garbage value
            for (int i = 0; i < nMagbins; i++){
                for (int j = 0; j < nThetabins; j++){
   
                    histogram[i][j] = 0;
                }
            }
            // Get histogram count based on each pixel
            for (int row = row_start; row < row_end; ++row){
                
                cv::Vec2b *row_ptr = image.ptr<cv::Vec2b>(row);
                for (int col = col_start; col < col_end; ++col){
                    int  mag_idx, theta_idx;

                    mag_idx = row_ptr[col][0]/ mag_bin_size;
                    theta_idx = row_ptr[col][1]/ theta_bin_size;

                    histogram[mag_idx][theta_idx] += 1; // Increment bin value by 1
                }
            }
            for (int i = 0; i < nMagbins; i++){
                for (int j = 0; j < nThetabins; j++){
                        fv.push_back((float)(histogram[i][j])/(pixel_count));
                    }
                }
            break;
        }
    }
    return 0;
}




int hist_feature_extraction(FeatureCfg *cfg, uint8_t size_type){

    ImageDirectoryAccess img_dir;
    img_dir.rewrite = 1;
    int loop_count = 0;
     // open the directory
    img_dir.dirp = opendir( cfg->databaseFolder );
    if( img_dir.dirp == NULL) {
        printf("Cannot open directory %s\n", cfg->databaseFolder);
        exit(-1);
    }

    int count = 0;
    // loop over all the files in the image file listing
    while( (img_dir.dp = readdir(img_dir.dirp)) != NULL ) {
        
    // check if the file is an image
        if( strstr(img_dir.dp->d_name, ".jpg") ||
            strstr(img_dir.dp->d_name, ".jpeg") ||            
            strstr(img_dir.dp->d_name, ".png") ||
            strstr(img_dir.dp->d_name, ".ppm") ||
            strstr(img_dir.dp->d_name, ".tif") ) 
            {

                loop_count += 1;
                if (loop_count>1){
                    img_dir.rewrite = 0;
                }    
                printf("processing image file: %s\n", img_dir.dp->d_name);

                // build the overall filename
                strcpy(img_dir.imageFullPath, cfg->databaseFolder);
                strcat(img_dir.imageFullPath, "/");
                strcat(img_dir.imageFullPath, img_dir.dp->d_name);
                strcpy(img_dir.imageFileName, img_dir.dp->d_name);

                //compute feature vector
                std::vector<float>feature_vector;

                cv::Mat img = cv::imread(img_dir.imageFullPath);

                if (cfg->mode == kModeTextureHOGMMatching){
                    cv::Mat grad_x, grad_y, grad, theta;
                    sobelX3x3(img, grad_x);
                    sobelY3x3(img, grad_y);
                    
                    if (cfg->type == kRGBHistogram){
                        magnitude(grad_x, grad_y, grad, theta);
                        get_hist_feature(grad, cfg->type, feature_vector, size_type);
                    }else if (cfg->type == kHOGHistogram){
                        magnitude_grayscale(grad_x, grad_y, grad);
                        get_hist_feature(grad, cfg->type, feature_vector, size_type);
                    } else if(cfg->type == kLAWHistogram){
                        get_law_feature(img_dir.imageFullPath, feature_vector);
                    }
                    
                } else if(cfg->mode == kModeLawFeatureMatching){
                    get_law_feature(img_dir.imageFullPath, feature_vector);

                } else {
                    get_hist_feature(img, cfg->type, feature_vector, size_type);
                }
                
                printf("Appending Image Feature Vector... %d\n", count);
                append_image_data_csv(cfg->csvFile,img_dir.imageFileName,feature_vector,img_dir.rewrite);
                count+=1;
            }

    }
    return 0;
}

int single_hist_feature_extraction(FeatureCfg *cfg){
    
    FeatureCfg new_cfg;
    new_cfg.databaseFolder = cfg->databaseFolder;
    new_cfg.mode = cfg->mode;
    new_cfg.csvFile = cfg->csvFile;
    new_cfg.type = cfg->type;
    hist_feature_extraction(&new_cfg, 1);
    printf("Finished Appending feature to CSV file");

    return 0;
}

int multi_hist_feature_extraction(FeatureCfg *cfg){

    FeatureCfg new_cfg;
    new_cfg.databaseFolder = cfg->databaseFolder;
    new_cfg.mode = cfg->mode;
    new_cfg.type = cfg->type;
    new_cfg.csvFile = cfg->csvFile;
    hist_feature_extraction(&new_cfg, 1); //Call hist feature extraction with first file name and full image size 
    new_cfg.csvFile = cfg->csvFile2;
    hist_feature_extraction(&new_cfg, 2); //Call hist feature extraction with second file name and other image size like half or custom size

    return 0;
}

int extract_texture_feature(FeatureCfg *cfg) {

    FeatureCfg new_cfg;
    new_cfg.databaseFolder = cfg->databaseFolder;
    new_cfg.mode = kModeHistogramMatching;
    new_cfg.type = kRGBHistogram;
    new_cfg.csvFile = cfg->csvFile;
    hist_feature_extraction(&new_cfg, 1); //Call hist feature extraction with first file name and full image size 
    new_cfg.mode = kModeTextureHOGMMatching;
    new_cfg.type = cfg->type;
    new_cfg.csvFile = cfg->csvFile2;
    hist_feature_extraction(&new_cfg, 1); //Call hist feature extraction with second file name and other image size like half or custom size    

    return 0;
}

int law_texture_feature(FeatureCfg *cfg) {

    FeatureCfg new_cfg;
    new_cfg.databaseFolder = cfg->databaseFolder;
    new_cfg.mode = kModeHistogramMatching;
    new_cfg.type = kRGBHistogram;
    new_cfg.csvFile = cfg->csvFile;
    hist_feature_extraction(&new_cfg, 1); //Call hist feature extraction with first file name and full image size 
    new_cfg.mode = kModeLawFeatureMatching;
    new_cfg.type = cfg->type;
    new_cfg.csvFile = cfg->csvFile2;
    hist_feature_extraction(&new_cfg, 1); //Call hist feature extraction with second file name and other image size like half or custom size    
    return 0;
}

int single_hist_compare(MatchCfg *cfg, std::vector<char *> &match_images){
    MatchCfg new_cfg;
    new_cfg.mode = cfg->mode;
    new_cfg.csvFile = cfg->csvFile;
    new_cfg.targetImageFile = cfg->targetImageFile;
    new_cfg.type = cfg->type;
    
    std::vector<float> dist_metric;
    std::vector<char *> images;
    std::map<double, char*> target_map;

    compare_histogram_features(&new_cfg, dist_metric, images ,1);

    for (int i=0; i<images.size(); i++){
        target_map[dist_metric[i]] = images[i];
    }
    
    printf("Computation Finished... \n");
    printf("Top %d matches for Target Image %s --> \n",cfg->N, cfg->targetImageFile);
    int count = 0;
    for (auto it = target_map.begin(); it != target_map.end() && count < cfg->N; ++it, ++count) {
        std::cout << it->first << ": " << it->second << std::endl;
        match_images.push_back(it->second);
    }

    return 0;
}

int multi_hist_compare(MatchCfg *cfg, std::vector<char *> &match_images){
    MatchCfg new_cfg;
    new_cfg.mode = cfg->mode;
    new_cfg.csvFile = cfg->csvFile;
    new_cfg.targetImageFile = cfg->targetImageFile;
    new_cfg.type = cfg->type;
    
    std::vector<float> dist_metric_1;
    std::vector<float> dist_metric_2;
    std::vector<char *> images_1;
    std::vector<char *> images_2;
    std::map<double, char*> target_map;
    
    compare_histogram_features(&new_cfg, dist_metric_1, images_1 ,1);
    new_cfg.csvFile = cfg->csvFile2;
    compare_histogram_features(&new_cfg, dist_metric_2, images_2 ,2);

    for (int i=0; i<images_1.size(); i++){
        float dist = dist_metric_1[i] * cfg->weights[0] + dist_metric_2[i] * cfg->weights[1];
        target_map[dist] = images_1[i];
    }
    
    printf("Computation Finished... \n");
    printf("Top %d matches for Target Image %s --> \n",cfg->N, cfg->targetImageFile);
    int count = 0;
    for (auto it = target_map.begin(); it != target_map.end() && count < cfg->N; ++it, ++count) {
        std::cout << it->first << ": " << it->second << std::endl;
        match_images.push_back(it->second);
    }

    return 0;

}


int texture_color_hist_compare(MatchCfg *cfg, std::vector<char *> &match_images){
    
    
    std::vector<float> dist_metric_1;
    std::vector<float> dist_metric_2;
    std::vector<char *> images_1;
    std::vector<char *> images_2;
    std::map<double, char*> target_map;
    
    MatchCfg new_cfg;
    new_cfg.mode = kModeHistogramMatching;
    new_cfg.csvFile = cfg->csvFile;
    new_cfg.targetImageFile = cfg->targetImageFile;
    new_cfg.type = kRGBHistogram;
    compare_histogram_features(&new_cfg, dist_metric_1, images_1 ,1); //Get RGB Colour Histogram for first feature

    new_cfg.mode = cfg->mode;
    new_cfg.csvFile = cfg->csvFile2;
    new_cfg.type = cfg->type;
    compare_histogram_features(&new_cfg, dist_metric_2, images_2 ,1); // Get texture feature for 2nd as per user configuration

    for (int i=0; i<images_1.size(); i++){
        float dist = dist_metric_1[i] * cfg->weights[0] + dist_metric_2[i] * cfg->weights[1];
        target_map[dist] = images_1[i];
    }
    
    printf("Computation Finished... \n");
    printf("Top %d matches for Target Image %s --> \n",cfg->N, cfg->targetImageFile);
    int count = 0;
    for (auto it = target_map.begin(); it != target_map.end() && count < cfg->N; ++it, ++count) {
        std::cout << it->first << ": " << it->second << std::endl;
        match_images.push_back(it->second);
    }
    return 0;
}


int law_texture_hist_compare(MatchCfg *cfg, std::vector<char *> &match_images){

    std::vector<float> dist_metric_1;
    std::vector<float> dist_metric_2;
    std::vector<char *> images_1;
    std::vector<char *> images_2;
    std::map<double, char*> target_map;
    
    MatchCfg new_cfg;
    new_cfg.mode = kModeHistogramMatching;
    new_cfg.csvFile = cfg->csvFile;
    new_cfg.targetImageFile = cfg->targetImageFile;
    new_cfg.type = kRGBHistogram;
    compare_histogram_features(&new_cfg, dist_metric_1, images_1 ,1); //Get RGB Colour Histogram for first feature

    new_cfg.mode = cfg->mode;
    new_cfg.csvFile = cfg->csvFile2;
    new_cfg.type = cfg->type;
    compare_histogram_features(&new_cfg, dist_metric_2, images_2 ,1); // Get texture feature for 2nd as per user configuration

    for (int i=0; i<images_1.size(); i++){
        float dist = dist_metric_1[i] * cfg->weights[0] + dist_metric_2[i] * cfg->weights[1];
        target_map[dist] = images_1[i];
    }
    
    printf("Computation Finished... \n");
    printf("Top %d matches for Target Image %s --> \n",cfg->N, cfg->targetImageFile);
    int count = 0;
    for (auto it = target_map.begin(); it != target_map.end() && count < cfg->N; ++it, ++count) {
        std::cout << it->first << ": " << it->second << std::endl;
        match_images.push_back(it->second);
    }
    return 0;
}


int compare_histogram_features(MatchCfg *cfg,std::vector <float> &distance_metric , std::vector <char *> &images,  uint8_t size_type){

    //compute feature vector for target image
    std::vector<float>target_fv;

    //read image database and feature vectors from csv file
    std::vector<std::vector<float> > feature_set;
    cv::Mat target_img = cv::imread(cfg->targetImageFile);

    if (cfg->mode == kModeTextureHOGMMatching){
        cv::Mat grad_x, grad_y, grad, theta;
        sobelX3x3(target_img, grad_x);
        sobelY3x3(target_img, grad_y);
        if(cfg->type == kRGBHistogram){
            magnitude(grad_x, grad_y, grad, theta);
            get_hist_feature(grad, cfg->type, target_fv, size_type);

        } else if(cfg->type == kHOGHistogram){ 
            magnitude_grayscale(grad_x, grad_y, grad);
            get_hist_feature(grad, cfg->type, target_fv, size_type);

        } else if(cfg->type == kLAWHistogram){
            get_law_feature(cfg->targetImageFile, target_fv);
        }

    } else if (cfg->mode == kModeLawFeatureMatching){
        get_law_feature(cfg->targetImageFile, target_fv);

    } else {
        get_hist_feature(target_img, cfg->type, target_fv, size_type);
    }
    read_image_data_csv(cfg->csvFile,images,feature_set,0);

    printf("Computing Distance Metric... \n");
    
    for(int i=0; i< images.size(); i++){
        if(cfg->type == kLAWHistogram){
           
            distance_metric.push_back(get_law_histogram_intersection(target_fv,feature_set[i]));
        }else {
            distance_metric.push_back(get_histogram_instersection(target_fv,feature_set[i]));
        }
    }
    return 0;
}

float get_histogram_instersection(std::vector<float> &fv1, 
                                   std::vector<float> &fv2){    
    // Calculate the sum of minimum values of intersection in each bin between two image histograms
    // Obtain weighted average of sums of multihistograms

    double sum = 0.0;
    for (int i=0; i < fv1.size(); i++){
        
        sum = sum + (fv1[i] >= fv2[i] ? fv2[i] : fv1[i]);
        // sum = sum + std::min(fv1[i], fv2[i]);  //intersection of corresponding histogram bins
    }
    
    return (1-sum);
}

float get_law_histogram_intersection(std::vector<float> &fv1, 
                                   std::vector<float> &fv2){    
    // Calculate the sum of minimum values of intersection in each bin between two image histograms
    // Obtain weighted average of sums of multihistograms

    float sum = 0.0;
    int count = 0;
    for (int i=0; i < fv1.size(); i++){
        sum = sum + (fv1[i] >= fv2[i] ? fv2[i] : fv1[i]);  //intersection of corresponding histogram bins
        count+=1;
        if(count==14){
            count = 0;
            sum=sum/14;
        }
    }
    return (1-sum);
}


int extract_law_feature(FeatureCfg *cfg){

    ImageDirectoryAccess img_dir;
    img_dir.rewrite = 1;
    int loop_count = 0;
     // open the directory
    img_dir.dirp = opendir( cfg->databaseFolder );
    if( img_dir.dirp == NULL) {
        printf("Cannot open directory %s\n", cfg->databaseFolder);
        exit(-1);
    }

    // loop over all the files in the image file listing
    while( (img_dir.dp = readdir(img_dir.dirp)) != NULL ) {
        
    // check if the file is an image
        if( strstr(img_dir.dp->d_name, ".jpg") ||
            strstr(img_dir.dp->d_name, ".jpeg") || 
            strstr(img_dir.dp->d_name, ".png") ||
            strstr(img_dir.dp->d_name, ".ppm") ||
            strstr(img_dir.dp->d_name, ".tif") ) 
            {

                loop_count += 1;
                if (loop_count>1){
                    img_dir.rewrite = 0;
                }    
                printf("processing image file: %s\n", img_dir.dp->d_name);

                // build the overall filename
                strcpy(img_dir.imageFullPath, cfg->databaseFolder);
                strcat(img_dir.imageFullPath, "/");
                strcat(img_dir.imageFullPath, img_dir.dp->d_name);
                strcpy(img_dir.imageFileName, img_dir.dp->d_name);

                //compute feature vector
                std::vector<float>feature_vector;
                get_law_feature(img_dir.imageFullPath,feature_vector);

                printf("Appending Image Feature Vector... %d\n", loop_count);
                append_image_data_csv(cfg->csvFile,img_dir.imageFileName,feature_vector,img_dir.rewrite);
                //printf("full path name: %s\n", img_dir.buffer);
        }

    }
    return 0;
}

int compare_law_features(MatchCfg *cfg, std::vector<char *> &match_images){

    //compute target feature vector
    std::vector<float>target_feature_vector;
    get_law_feature(cfg->targetImageFile,target_feature_vector);    

    //read image database and feature vectors from csv file
    std::vector<std::vector<float> > feature_set;
    std::vector<char *> images;
    read_image_data_csv(cfg->csvFile,images,feature_set,0);

    //store distance metric values for each image wrt target image
    std::map<float, char*> target_map;

    printf("Computing Distance Metric... \n");
    for(int i=0; i< images.size(); i++){

        float bs_metric = get_law_histogram_intersection(target_feature_vector,feature_set[i]);
        target_map[bs_metric] = images[i];
    }

    printf("Top %d matches for Target Image %s ::: \n",cfg->N,cfg->targetImageFile);
    int count = 0;
    for (auto it = target_map.begin(); it != target_map.end() && count < cfg->N; ++it, ++count) {
        std::cout << it->first << ": " << it->second << std::endl;
        match_images.push_back(it->second);
    }
    
    return 0;
}


int get_law_feature(char *image_path, std::vector<float> &feature_vector){

    // Load an image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);


    // Define law  filter kernels
 
    cv::Mat l5e5RT = (cv::Mat_<float>(5,5) <<        -1   , -3    ,-3  ,  -1 ,    0,
                                            -3,    -8  ,  -6 ,    0,     1,
                                            -3   , -6   ,  0  ,   6   ,  3,
                                              -1  ,   0,     6  ,   8  ,   3,
                                             0,    1  ,   3   ,  3  ,   1);


    cv::Mat l5s5RT = (cv::Mat_<float>(5,5) <<       -1 ,   -2  ,  -2   , -2  ,  -1,
                                            -2   ,  0    , 4     ,0   , -2,
                                            -2  ,   4  ,  12 ,    4  ,  -2,
                                             -2  ,   0   ,  4  ,   0  ,  -2,
                                            -1 ,   -2   , -2 ,   -2,    -1);

    cv::Mat l5w5RT = (cv::Mat_<float>(5,5) <<       -1  ,  -1,    -3  ,  -3   ,  0,
                                            -1  ,   8,    6  ,   0   ,  3,
                                            -3  ,   6  ,   0 ,   -6   ,  3,
                                              -3 ,    0  ,  -6 ,   -8 ,    1,
                                             0 ,    3  ,   3 ,    1 ,    1);

    cv::Mat l5r5RT = (cv::Mat_<float>(5,5) <<        1,     0 ,    6 ,    0  ,   1,
                                            0 ,  -16,     0 ,  -16  ,   0,
                                            6   ,  0  ,  36     ,0   ,  6,
                                             0 ,  -16 ,    0   ,-16   ,  0,
                                           1  ,   0  ,   6  ,   0  ,   1);

    cv::Mat e5s5RT = (cv::Mat_<float>(5,5) <<       1 ,    1 ,   -1 ,   -1  ,   0,
                                            1  ,   0 ,   -2,     0   ,  1,
                                           -1 ,   -2 ,    0  ,   2  ,   1,
                                            -1 ,    0  ,   2 ,    0   , -1,
                                             0,     1  ,   1 ,   -1  ,  -1);

    cv::Mat e5w5RT = (cv::Mat_<float>(5,5) <<        1 ,    0 ,    0,     0 ,   -1,
                                            0  ,  -4  ,   0 ,    4   ,  0,
                                             0  ,   0   ,  0   ,  0  ,   0,
                                           0    , 4  ,   0 ,   -4   ,  0,
                                            -1   ,  0   ,  0  ,   0  ,   1);

    cv::Mat e5r5RT = (cv::Mat_<float>(5,5) <<        -1,     1  ,  -3 ,    3  ,   0,
                                            1  ,   8 ,   -6   ,  0,   -3,
                                             -3  ,  -6 ,    0     ,6   ,  3,
                                              3 ,    0,     6,    -8 ,   -1,
                                             0   , -3 ,   3  ,  -1 ,    1);

    cv::Mat s5w5RT = (cv::Mat_<float>(5,5) <<        1  ,  -1 ,   -1  ,   1  ,   0,
                                           -1 ,    0  ,   2  ,   0    ,-1,
                                           -1 ,    2 ,    0   , -2   ,  1,
                                           1  ,   0 ,   -2  ,   0   ,  1,
                                            0  ,  -1 ,    1 ,    1  ,  -1);

    cv::Mat s5r5RT = (cv::Mat_<float>(5,5) <<       -1 ,    2 ,   -2 ,    2 ,   -1,
                                            2   ,  0   , -4  ,   0   ,  2,
                                           -2  ,  -4   , 12  ,  -4 ,   -2,
                                            2 ,    0  ,  -4    , 0 ,    2,
                                           -1   ,  2    ,-2    , 2  ,  -1);
    
    cv::Mat w5r5RT = (cv::Mat_<float>(5,5) <<        -1   ,  3  ,  -3  ,   1 ,    0,
                                            3 ,   -8   ,  6   ,  0  ,  -1,
                                            -3  ,   6  ,   0    ,-6   ,  3,
                                            1  ,   0 ,   -6 ,    8  ,  -3,
                                             0  ,  -1 ,    3 ,   -3    , 1);


    cv::Mat e5e5RT = (cv::Mat_<float>(5,5) <<        1 ,    2  ,   0 ,   -2,    -1,
                                            2 ,    4 ,    0,    -4  ,  -2,
                                            0  ,   0 ,    0    , 0    , 0,
                                            -2   , -4  ,   0  ,   4  ,   2,
                                            -1 ,   -2  ,   0  ,   2   ,  1);

    cv::Mat s5s5RT = (cv::Mat_<float>(5,5) <<         1 ,    0   , -2 ,    0 ,    1,
                                                0 ,    0  ,   0 ,    0 ,    0,
                                                -2 ,    0 ,    4  ,   0  ,  -2,
                                                0  ,   0  ,  0    , 0    , 0,
                                                1 ,    0 ,   -2 ,    0,     1);

    cv::Mat w5w5RT = (cv::Mat_<float>(5,5) <<         1  ,  -2 ,    0 ,    2 ,   -1,
                                                -2 ,    4 ,    0  ,  -4 ,    2,
                                                0,     0     ,0     ,0  ,   0,
                                                2,    -4   ,  0    , 4    ,-2,
                                                -1  ,   2  ,   0  ,  -2  ,   1);

    cv::Mat r5r5RT = (cv::Mat_<float>(5,5) <<          1   , -4  ,   6 ,   -4 ,    1,
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
    int num_of_bins = 16;
    int bin_size = 256/num_of_bins;
    for (int i = 0; i < num_kernels;i++){

        cv::Mat filtered_image;
        filter2D(image, filtered_image, CV_16SC1, law_kernels[i], cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
        cv::convertScaleAbs(filtered_image,filtered_image,1);

        float pixel_count = filtered_image.rows * filtered_image.cols;
        float histogram[num_of_bins];
        for (int count = 0; count < num_of_bins; ++count){
            histogram[count] = 0;
        }
        for (int row=0; row<filtered_image.rows; ++row){

            uchar *row_ptr = filtered_image.ptr<uchar>(row);

            for (int col=0; col<filtered_image.cols; ++col){
                int bin_index = row_ptr[col]/bin_size;
                histogram[bin_index] +=1;
            }
        }

        for (int count =0; count<num_of_bins; ++count){
            feature_vector.push_back(histogram[count]/pixel_count);
        }
    }
    return 0;
}
   

int extract_spatial_var_feature(FeatureCfg *cfg){

    ImageDirectoryAccess img_dir;
    img_dir.rewrite = 1;
    int loop_count = 0;
     // open the directory
    img_dir.dirp = opendir( cfg->databaseFolder );
    if( img_dir.dirp == NULL) {
        printf("Cannot open directory %s\n", cfg->databaseFolder);
        exit(-1);
    }

    // loop over all the files in the image file listing
    while( (img_dir.dp = readdir(img_dir.dirp)) != NULL ) {
        
    // check if the file is an image
        if( strstr(img_dir.dp->d_name, ".jpg") ||
            strstr(img_dir.dp->d_name, ".jpeg") || 
            strstr(img_dir.dp->d_name, ".png") ||
            strstr(img_dir.dp->d_name, ".ppm") ||
            strstr(img_dir.dp->d_name, ".tif") ) 
            {

                loop_count += 1;
                if (loop_count>1){
                    img_dir.rewrite = 0;
                }    
                printf("processing image file: %s\n", img_dir.dp->d_name);

                // build the overall filename
                strcpy(img_dir.imageFullPath, cfg->databaseFolder);
                strcat(img_dir.imageFullPath, "/");
                strcat(img_dir.imageFullPath, img_dir.dp->d_name);
                strcpy(img_dir.imageFileName, img_dir.dp->d_name);

                //compute feature vector
                std::vector<float>feature_vector;
                get_spatial_var_feature(img_dir.imageFullPath,feature_vector);

                printf("Appending Image Feature Vector... %d\n", loop_count);
                append_image_data_csv(cfg->csvFile,img_dir.imageFileName,feature_vector,img_dir.rewrite);
                //printf("full path name: %s\n", img_dir.buffer);
        }

    }
    return 0;
}

int get_spatial_var_feature(char *image_path, std::vector<float> &feature_vector){

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    // Convert the image to the HSV color space
    cv::Mat hsv;
    cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Define the number of bins for each channel
    int hbins = 30, sbins = 32, vbins = 32;

    // Define the range of each channel
    float hranges[] = { 0, 180 };
    float sranges[] = { 0, 256 };
    float vranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges, vranges };

    // Split the image into separate color channels
    cv::Mat h, s, v;
    std::vector<cv::Mat> hsv_channels;
    split(hsv, hsv_channels);
    h = hsv_channels[0];
    s = hsv_channels[1];
    v = hsv_channels[2];


    // Compute the spatial variance of each color bin in the H, S, and V channels
    cv::Mat hist;
    float mean[3] = {0}, stddev[3] = {0};
    for (int i = 0; i < 3; i++) {
        cv::Mat channel = (i == 0) ? h : (i == 1) ? s : v;
        int channels[] = { 0 };
        int dims = 1;
        float pixel_count = channel.rows * channel.cols;
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

        calcHist(&channel, 1, channels, cv::Mat(), hist, dims, &hbins, ranges, true, false);

        // Normalize the histogram
        // normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        // Compute the mean and standard deviation of the histogram

        
        // Compute the mean and standard deviation of the histogram
        
        
        for (int j = 0; j < hist.rows; j++) {
            float temp = hist.at<float>(j, 0)/pixel_count;
            mean[i] += temp * j;
            stddev[i] += temp * pow(j - mean[i], 2);
        }
        stddev[i] = sqrt(stddev[i] / hist.rows);
        // cout << "Channel " << i << ": mean = " << mean[i] << ", stddev = " << stddev[i] << endl;
    }

    
    //////////////////////////////////////////////////
    int h_bins = 8;
    int s_bins = 8;
    int v_bins = 8;

    // Set the number of bins for each channel in an array
    int histSize[] = { h_bins, s_bins, v_bins };

    // Compute the histogram
    cv::Mat hist_op(8, 8, 8, CV_32F);
    int channels[] = { 0, 1, 2 };
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist_op, 3, histSize, ranges, true, false);

    // Normalize the histogram
    // cv::normalize(hist_op, hist_op, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());


    float pixel_count = hsv.rows * hsv.cols;
    // std::cout << "pixel_count: " << (float)pixel_count << std::endl;
    int count = 0;
    for (int x = 0; x < hist_op.size[0]; x++){
        for (int y = 0; y < hist_op.size[1]; y++){
            for (int z = 0; z < hist_op.size[2]; z++){
                //count++;
                feature_vector.push_back((hist_op.at<float>(x, y, z))/(pixel_count));
                
            }
        }
    }
    feature_vector.push_back(stddev[0]);
    feature_vector.push_back(stddev[1]);
    feature_vector.push_back(stddev[2]);

    return 0;
}


int compare_spatial_var_features(MatchCfg *cfg, std::vector<char *> &match_images){

    //compute target feature vector
    std::vector<float>target_feature_vector;
    get_spatial_var_feature(cfg->targetImageFile,target_feature_vector);    

    //read image database and feature vectors from csv file
    std::vector<std::vector<float>> feature_set;
    std::vector<char *> images;
    read_image_data_csv(cfg->csvFile,images,feature_set,0);

    //store distance metric values for each image wrt target image
    std::map<float, char*> target_map;

    printf("Computing Distance Metric... \n");
    for(int i=0; i< images.size(); i++){
        
        float hist_metric = get_spatial_var_histogram_intersection(target_feature_vector,feature_set[i]);
        target_map[hist_metric] = images[i];
    }

    printf("Top %d matches for Target Image %s ::: \n",cfg->N,cfg->targetImageFile);
    int count = 0;
    for (auto it = target_map.begin(); it != target_map.end() && count < cfg->N; ++it, ++count) {
        std::cout << it->first << ": " << it->second << std::endl;
        match_images.push_back(it->second);
    }
    
    return 0;
}


float get_spatial_var_histogram_intersection(std::vector<float> &fv1, std::vector<float> &fv2){

    // Calculate the sum of minimum values of intersection in each bin between two image histograms
    // Obtain weighted average of sums of multihistograms

    double sum1 = 0.0;

    int size_of_fv = fv1.size();
    for (int i=0; i < fv1.size()-3; i++){
        
        sum1 = sum1 + (fv1[i] >= fv2[i] ? fv2[i] : fv1[i]);

        // sum = sum + std::min(fv1[i], fv2[i]);  //intersection of corresponding histogram bins
    }
    sum1 = 1 - sum1;

    double sum2 = 0.0;
    for (int i = (size_of_fv -3); i < size_of_fv; i++ ){
        sum2 += (fv1[i] - fv2[i]) * (fv1[i] - fv2[i]);
    }
    
    return sqrt(0.5*sum1 + 0.5* sum2);
}