#include <dirent.h>

#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H


typedef enum {

    kModeBaselineMatching,
    kModeHistogramMatching,
    kModeMultiHistogramMatching,
    kModeTextureHOGMMatching,
    kModeLawFeatureMatching,
    kModeSpatialVarMatching,

}FeatureDetectionMode;

typedef struct {

    char imageFileName[100];
    char imageFullPath[255];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int rewrite;
} ImageDirectoryAccess;


typedef enum {
    kRGHistogram,
    kGBHistogram,
    kRBHistogram,
    kRGBHistogram,
    kHOGHistogram,
    kLAWHistogram
} HistogramType;

typedef struct {
    char *databaseFolder;
    char *csvFile;
    char *csvFile2; //Use for multihistogram
    FeatureDetectionMode mode;
    HistogramType type;
} FeatureCfg;

typedef struct {
    int N;
    char *csvFile;
    char *csvFile2; //Use for multihistorgram
    char *targetImageFile;
    float *weights;
    FeatureDetectionMode mode;
    HistogramType type;
} MatchCfg;

int save_feature_set(FeatureCfg *cfg);
int match_feature_set(MatchCfg *cfg, std::vector<char *> &match_images);

//Baseline matching functions
int get_baseline_feature(char *image, std::vector<float> &feature_vector);
int baseline_feature_extraction(FeatureCfg *cfg);
int compare_baseline_feature(MatchCfg *cfg, std::vector<char *> &match_images);
float get_baseline_distance_metric(std::vector<float> &fv1, std::vector<float> &fv2);

//MultiHistogram Matching functions
int single_hist_feature_extraction(FeatureCfg *cfg);
int multi_hist_feature_extraction(FeatureCfg *cfg);
int get_hist_feature(cv::Mat &image, HistogramType type, std::vector<float> &fv, uint8_t size_to_consider);
int hist_feature_extraction(FeatureCfg *cfg, uint8_t size_type);
int single_hist_compare(MatchCfg *cfg, std::vector<char *> &match_images);
int multi_hist_compare(MatchCfg *cfg, std::vector<char *> &match_images);
int compare_histogram_features(MatchCfg *cfg,std::vector <float> &distance_metric , std::vector <char *> &images,  uint8_t size_type);
float get_histogram_instersection(std::vector<float> &fv1, std::vector<float> &fv2);


int extract_texture_feature(FeatureCfg *cfg);
int texture_color_hist_compare(MatchCfg *cfg, std::vector<char *> &match_images);

int extract_law_feature(FeatureCfg *cfg);
int compare_law_features(MatchCfg *cfg, std::vector<char *> &match_images);
int get_law_feature(char *image_path, std::vector<float> &feature_vector);

float get_law_histogram_intersection(std::vector<float> &fv1, std::vector<float> &fv2);

int extract_spatial_var_feature(FeatureCfg *cfg);
int get_spatial_var_feature(char *image_path, std::vector<float> &feature_vector);
int compare_spatial_var_features(MatchCfg *cfg, std::vector<char *> &match_images);
float get_spatial_var_histogram_intersection(std::vector<float> &fv1, std::vector<float> &fv2);



///Extension 1
int law_texture_feature(FeatureCfg *cfg);
int law_texture_hist_compare(MatchCfg *cfg, std::vector<char *> &match_images);

#endif 