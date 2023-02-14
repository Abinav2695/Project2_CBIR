cmake_minimum_required(VERSION 3.1)

project(ContentBasedImageRetrieval VERSION 0.1.0
              LANGUAGES CXX
              HOMEPAGE_URL "https://github.com/Abinav2695/Pattern_Recognition_CV/tree/main/Real_Time_Filtering")


set(PROJECT_SRC_DIR src)
set(PROJECT_INC_DIR include)
set(IMG_DISPLAY ${PROJECT_SRC_DIR}/imgDisplay.cpp)
set(VID_DISPLAY ${PROJECT_SRC_DIR}/vidDisplay.cpp)
set(FILTER_APPS ${PROJECT_SRC_DIR}/filter.cpp)
set(FEATURE_APPS ${PROJECT_SRC_DIR}/feature.cpp)

set(MAIN ${PROJECT_SRC_DIR}/main.cpp)
set(HOG ${PROJECT_SRC_DIR}/hog.cpp)
set(SPV ${PROJECT_SRC_DIR}/spatial_variance.cpp)
set(CSV_UTILS ${PROJECT_SRC_DIR}/csv_util.cpp)
set(TRAIN ${PROJECT_SRC_DIR}/train.cpp)
set(TEST ${PROJECT_SRC_DIR}/test.cpp)

set(PROJECT_LIBRARIES_DIR libs)
set(BIN_DIR bin)

#####################################################################
# DEPENDENCIES
#####################################################################

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${PROJECT_INC_DIR})


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})

# add_executable( imgDisplay ${IMG_DISPLAY} ${FILTER_APPS})
# add_executable( vidDisplay ${VID_DISPLAY} ${FILTER_APPS})
add_executable( main ${MAIN} ${CSV_UTILS} ${FEATURE_APPS} ${FILTER_APPS})
add_executable( train ${TRAIN} ${CSV_UTILS} ${FEATURE_APPS} ${FILTER_APPS})
add_executable( test ${TEST} ${CSV_UTILS} ${FEATURE_APPS} ${FILTER_APPS})
add_executable( hog ${HOG})
add_executable( spv ${SPV})

# target_link_libraries( imgDisplay ${OpenCV_LIBS} )
# target_link_libraries( vidDisplay ${OpenCV_LIBS} )
target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries( train ${OpenCV_LIBS} )
target_link_libraries( test ${OpenCV_LIBS} )
target_link_libraries( hog ${OpenCV_LIBS} )
target_link_libraries( spv ${OpenCV_LIBS} )