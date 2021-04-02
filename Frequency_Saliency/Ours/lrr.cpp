#include "lrr.h"

MatrixXd LowRankRepresentation::solve_l1l2(MatrixXd& W, double lambda)
{
	MatrixXd E;
	VectorXd temp;
	int n, i;

	n = W.cols();
	E = W;

	for (i = 0; i < n; i++)
	{
		temp = W.col(i);
		E.col(i) = solve_l2(temp, lambda);
	}
	return E;
}

VectorXd LowRankRepresentation::solve_l2(VectorXd& w, double lambda)
{
	double nw;
	VectorXd x;

	nw = w.norm();
	if (nw > lambda)
		x = (nw - lambda) * w / nw;
	else
		x = VectorXd::Zero(w.size(), 1);
	return x;
}


std::vector<MatrixXd> LowRankRepresentation::result(MatrixXd& X, double lambda)
{
	int d = X.rows();
	int n = X.cols();

	MatrixXd xtx = X.transpose() * X;
	MatrixXd inv_x = (xtx + MatrixXd::Identity(n, n)).inverse();
	MatrixXd J = MatrixXd::Zero(n, n);
	MatrixXd Z = MatrixXd::Zero(n, n);
	MatrixXd E = MatrixXd::Zero(d, n);

	MatrixXd Y1 = MatrixXd::Zero(d, n);
	MatrixXd Y2 = MatrixXd::Zero(n, n);

	int iter = 0;
	FullPivLU<MatrixXd> lu_decomp(Z);
	//std::cout << "initial, rank=" << lu_decomp.rank() << std::endl;
	while (iter < MaxIter)
	{
		iter += 1;
		MatrixXd tmp = Z + Y2 / Mu;
		JacobiSVD<MatrixXd> svd(tmp, ComputeThinU | ComputeThinV);
		VectorXd sigma = svd.singularValues();
		//sigma = MatrixXd(sigma.asDiagonal());
		MatrixXd U = svd.matrixU();
		MatrixXd V = svd.matrixV();
		int svp = (sigma.array() > 1 / Mu).count();

		if (svp >= 1)
		{
			VectorXd sigma_tmp = sigma.segment(0, svp) - (1 / Mu) * VectorXd::Ones(svp);
			sigma = sigma_tmp;
		}
		else
		{
			svp = 1;
			sigma = VectorXd::Zero(1);
		}
		J = U.leftCols(svp) * MatrixXd(sigma.asDiagonal()) * V.leftCols(svp).transpose();

		Z = inv_x * (xtx - X.transpose() * E + J + (X.transpose() * Y1 - Y2) / Mu);

		MatrixXd xmaz = X - X * Z;
		tmp = xmaz + Y1 / Mu;
		E = solve_l1l2(tmp, lambda / Mu);

		MatrixXd leq1 = xmaz - E;
		MatrixXd leq2 = Z - J;
		double stopC_tmp1 = leq1.array().abs().maxCoeff();
		double stopC_tmp2 = leq2.array().abs().maxCoeff();
		double stopC = stopC_tmp1 < stopC_tmp2 ? stopC_tmp2 : stopC_tmp1;

		if (iter == 1 || (iter % 50) == 0 || stopC < Tol)
		{
			FullPivLU<MatrixXd> lu_decomp(Z);
			//printf("iter %d, mu=%2.1e, rank=%d, stopALM=%2.3e\n", iter, Mu, lu_decomp.rank(), stopC);
			printf("iter %d, mu=%2.1e, stopALM=%2.3e\n", iter, Mu, stopC);
		}
		if (stopC < Tol)
		{
			std::cout << "LRR done." << std::endl;
			break;
		}
		else
		{
			Y1 = Y1 + Mu * leq1;
			Y2 = Y2 + Mu * leq2;
			Mu = MaxMu > Mu*Rho ? Mu * Rho : MaxMu;
		}
	}

	std::vector<MatrixXd> ZE;
	ZE.push_back(Z);
	ZE.push_back(E);
	return ZE;
}