#ifndef __LRR_H_
#define __LRR_H_

#include <stdio.h>
#include <iostream> 
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>

using namespace Eigen;

static const double Tol = 1e-8;
static const int MaxIter = 1e6;
static const double Rho = 1.1;

static double Mu = 1e-6;
static const long int MaxMu = 1e10;

//static const long int MaxMu = 1e30;
//static double Mu = 1e-2;

class LowRankRepresentation
{
public:
    std::vector<MatrixXd> result(MatrixXd& X, double lambda);
    MatrixXd solve_l1l2(MatrixXd& W, double lambda);
private:
    VectorXd solve_l2(VectorXd& w, double lambda);
};



#endif // !__LRR_H_
