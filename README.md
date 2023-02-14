# Time Travel Days used -- 3
# Project done by Abinav Anantharaman and Satwik Shridhar Bhandiwad


# CS 5330 Spring 2023 --> CBIR @ NEU

## Operating System and IDE:

I have used Ubuntu 20.04.5 LTS OS and Visual Studi Code 1.74.2 for this project. 
The IDE doesn't really matters because I have compiled and built the workspace using cmake and make terminal commands

### Installation of Dependencies
*  cmake,gcc and g++ installation
```bash
sudo apt install cmake gcc g++
```

### Workspace Installation
* Clone this workspace
```bash
cd [$Your_Destination_Folder]
git clone https://github.com/Abinav2695/Project2_CBIR.git
```
* Build workspace
```bash
cd Project2_CBIR/
mkdir build
cd build
cmake -S .. -B .
make
```

### Running Executables
* To build features  
```bash
cd build
./bin/train 
```

* To run image matching 
```bash
cd build
./bin/test 
```

## Usage
train usage:
Usage :  Enter the task number for training dataset   
 '1' : Task 1 --> Baseline Feature Set  
 '2' : Task 2 --> RG Histogram Feature Set  
 '3' : Task 2 --> RGB Histogram Feature Set 
 '4' : Task 3 --> RGB Multi Histogram Feature Set 
 '5' : Task 4 --> RGB Colour Histogram and Sobel Gradient Texture Feature Set: 
 '6' : Task 5 --> RGB Colour Histogram and Sobel Gradient Texture Feature Set on Custom Dataset  
 '7' : Extension 1 --> HSV colour Histogram and Spatial Variance Feature Set on Custom Dataset  
 '8' : Extension 2 --> RGB Colour Histogram and LAW Filter Texture Feature Set on Custom Dataset 
 '9' : To quit program: 

test Usage:
Usage :  Enter the task number for matching method with imagepath   
Example: 1 full_image_path/image.jpg   

 '1' {image_path} : Task 1 --> Baseline Feature Matching  
 '2' {image_path}: Task 2 --> RG Histogram Feature Set  
 '3' {image_path}: Task 2 --> RGB Histogram Feature Set 
 '4' {image_path}: Task 3 --> RGB Multi Histogram Feature Set 
 '5' {image_path}: Task 4 --> RGB Colour Histogram and Sobel Gradient Texture Feature Set: 
 '6' {image_path}: Task 5 --> RGB Colour Histogram and Sobel Gradient Texture Feature Set on Custom Dataset  
 '7' {image_path}: Extension 1 --> HSV colour Histogram and Spatial Variance Feature Set on Custom Dataset  
 '8' {image_path}: Extension 2 --> RGB Colour Histogram and LAW Filter Texture Feature Set on Custom Dataset 
 '9' : To quit program: 
 