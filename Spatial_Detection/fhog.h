#ifndef _FHOG_H_
#define _FHOG_H_

#include <stdio.h>
#include "opencv2/imgproc/imgproc_c.h"

// DataType: STRUCT featureMap
// FEATURE MAP DESCRIPTION
//   Rectangular map (sizeX x sizeY), 
//   every cell stores feature vector (dimension = numFeatures)
// map             - matrix of feature vectors
//                   to set and get feature vectors (i,j) 
//                   used formula map[(j * sizeX + i) * p + k], where
//                   k - component of feature vector in cell (i, j)
typedef struct{
	int sizeX;
	int sizeY;
	int numFeatures;
	float *map;
} CvLSVMFeatureMapCaskade;


//#include "float.h"

#define PI    CV_PI

// The number of elements in bin
// The number of sectors in gradient histogram building
#define NUM_SECTOR 9

/*
// Getting feature map for the selected subimage
//
// API
// int getFeatureMaps(const IplImage * image, const int k, featureMap **map);
// INPUT
// image             - selected subimage
// k                 - size of cells
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int getFeatureMaps(const IplImage * image, const int k, CvLSVMFeatureMapCaskade **map);


/*
// Feature map Normalization and Truncation
//
// API
// int normalizationAndTruncationFeatureMaps(featureMap *map, const float alfa);
// INPUT
// map               - feature map
// alfa              - truncation threshold
// OUTPUT
// map               - truncated and normalized feature map
// RESULT
// Error status
*/
int normalizeAndTruncate(CvLSVMFeatureMapCaskade *map, const float alfa);

/*
// Feature map reduction
// In each cell we reduce dimension of the feature vector
// according to original paper special procedure
//
// API
// int PCAFeatureMaps(featureMap *map)
// INPUT
// map               - feature map
// OUTPUT
// map               - feature map
// RESULT
// Error status
*/
int PCAFeatureMaps(CvLSVMFeatureMapCaskade *map);

int allocFeatureMapObject(CvLSVMFeatureMapCaskade **obj, const int sizeX, const int sizeY,
	const int p);

int freeFeatureMapObject(CvLSVMFeatureMapCaskade **obj);


#endif
