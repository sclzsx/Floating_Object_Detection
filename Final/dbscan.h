#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>

using namespace std;

struct dbscanPoint{
    double x;
    double y;
    int lable;  // -1 unvisited, 0 noise, >0 cluster index
};
double euclidean_distance(dbscanPoint a, dbscanPoint b);
int dbscan(vector<dbscanPoint> &dataset, double eps, int min_pts);

#endif /*DBSCAN_H*/
