#ifndef GLOBAL_VARIABLES_H_
#define GLOBAL_VARIABLES_H_

//#define train_flag_     1

#define fhog_type_		1
#define hog_type_		2
#define lbp_type_		3
#define glcm_type_		4
#define fhog_glcm_type_ 5

#define bms_type_		6
#define hc_type_		7
#define uav_type_		8
#define sr_type_		9
#define myphase_type_	10
#define myphase2_type_	11

#define svm_type_		12
#define bayes_type_		13
#define ann_type_		14
#define knn_type_		15
#define rtrees_type_	16
#define adaboost_type_  17

#include "feature.h"
#include "fhog.h"

#include "dbscan.h"
#include "process_rects.h"
#include "gen_features.h"
#include "unsupervised.h"


const int winsize = 80;

const int cellsize = 16;
const int stride = 20;

const string groundtruth_filename = "E:\\DataSets\\litter\\src_V2\\src_test.xml";
const string PosPath = "E:\\DataSets\\litter\\experiment_3_22\\train\\pos\\";
const string NegPath = "E:\\DataSets\\litter\\experiment_3_22\\train\\neg\\";
const string detestTestName = "E:\\DataSets\\litter\\experiment_3_22\\test\\img\\";

const string info = ".\\xml\\cell" + std::to_string(cellsize) + "_win" + std::to_string(winsize) + "_";

#endif