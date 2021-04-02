#include "unsupervised.h"

void MyGammaCorrection(Mat src, Mat& dst, float fGamma)
{
	CV_Assert(src.data);
	CV_Assert(src.depth() != sizeof(uchar));

	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{

		cv::MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			*it = lut[(*it)];

		break;
	}
	case 3:
	{

		cv::MatIterator_<cv::Vec3b> it, end;
		for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; it++)
		{
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}
}

void get_saliency_rects(Mat input, std::vector<cv::Rect> &output_rects)
{
	Mat saliency_in = input.clone();
	Mat saliency;


	get_myphase_saliency2(saliency_in, saliency);


	if (saliency.channels() != 1)
		cvtColor(saliency, saliency, cv::COLOR_RGB2GRAY);
	if (saliency.type() != CV_8UC1)
		saliency.convertTo(saliency, CV_8UC1);

	Mat bin_sr;
	threshold(saliency, bin_sr, 0, 255, cv::THRESH_OTSU);
	Mat element_sr = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
	morphologyEx(bin_sr, bin_sr, cv::MORPH_CLOSE, element_sr);
	morphologyEx(bin_sr, bin_sr, cv::MORPH_DILATE, element_sr);
	std::vector<std::vector<cv::Point> >contours_sr;
	std::vector<cv::Vec4i> hierarchy_sr;
	findContours(bin_sr, contours_sr, hierarchy_sr, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours_sr.size(); i++)
	{
		int areaTmp = contourArea(contours_sr[i]);
		if (areaTmp > 10 && areaTmp < 10000)
		{
			cv::Rect rectTmp = boundingRect(contours_sr[i]);
			output_rects.push_back(rectTmp);
		}
	}
}

void low_rank_decomposition(Mat input, Mat &z, Mat &e)
{
	MatrixXd mat(input.rows, input.cols);
	cv::cv2eigen(input, mat);

	double lambda = 0.01;
	LowRankRepresentation mylrr;
	std::vector<MatrixXd> ZE = mylrr.result(mat, lambda);

	cv::eigen2cv(ZE[0], z);
	cv::eigen2cv(ZE[1], e);
}

void get_phase_map(Mat img, Mat &output)
{
	//cv::Mat img = cv::imread("test.jpg");
	if (img.channels() != 1)
		cvtColor(img, img, cv::COLOR_RGB2GRAY);
	float ratio = 64.0 / img.cols;
	cv::resize(img, img, cv::Size(img.cols*ratio, img.rows*ratio));

	cv::Mat planes[] = { cv::Mat_<float>(img), cv::Mat::zeros(img.size(), CV_32F) };
	cv::Mat complexImg;
	cv::merge(planes, 2, complexImg);
	cv::dft(complexImg, complexImg);
	cv::split(complexImg, planes);

	//cv

	//cv::Mat mag, logmag, smooth, spectralResidual;
	//cv::magnitude(planes[0], planes[1], mag);
	//
	//cv::log(mag, logmag);
	//cv::boxFilter(logmag, smooth, -1, cv::Size(3, 3));
	//cv::subtract(logmag, smooth, spectralResidual);
	//cv::exp(spectralResidual, spectralResidual);

	//planes[0] = planes[0].mul(spectralResidual) / mag;
	//planes[1] = planes[1].mul(spectralResidual) / mag;

	//cv::merge(planes, 2, complexImg);
	//cv::dft(complexImg, complexImg, cv::DFT_INVERSE | cv::DFT_SCALE);
	//cv::split(complexImg, planes);
	//cv::magnitude(planes[0], planes[1], mag);
	//cv::multiply(mag, mag, mag);
	//cv::GaussianBlur(mag, mag, cv::Size(9, 9), 2.5, 2.5);
	//cv::normalize(mag, mag, 1.0, 0.0, cv::NORM_MINMAX);

	//output = mag.clone();
}

void get_myphase_saliency2(Mat input, Mat &output)
{
	Mat myphase_in = input.clone();

	MyGammaCorrection(myphase_in, myphase_in, 5);
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(myphase_in, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(myphase_in, grad_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, myphase_in);

	Mat chushi;
	get_phase_map(myphase_in, chushi);
	imshow("chushi", chushi);


	//Mat chushi_clone1 = chushi.clone();
	//cv::resize(chushi_clone1, chushi_clone1, cv::Size(chushi_clone1.cols/10, chushi_clone1.rows/10), cv::INTER_AREA);
	//Mat low_rank_part, sparse_part;
	//low_rank_decomposition(chushi_clone1, low_rank_part, sparse_part);
	//Mat quanju = sparse_part.clone();
	//cv::resize(quanju, quanju, input.size(), cv::INTER_CUBIC);
	//imshow("quanju", quanju);












	//int div = 10;
	//Mat chushi_clone2 = chushi.clone();
	//int blocknum = 0;
	//int blockwidth = chushi_clone2.cols/div;
	//int blockheight = chushi_clone2.rows/div;
	//int num = div*div;
	//std::cout << "chushi.size:"<< chushi_clone2.cols << " " << chushi_clone2.rows<< std::endl;
	//std::cout << "block.size:" << blockwidth << " " << blockheight << std::endl;
	//std::cout << "theory blocknum:" << num << std::endl;

	//Mat Y = Mat::zeros(blockheight, num, CV_64FC1);
	//std::vector<Mat> pk;
	//for (int y = 0; y < chushi_clone2.cols - blockwidth + 1; y = y + blockwidth)
	//{
	//	for (int x = 0; x < chushi_clone2.rows - blockheight + 1; x = x + blockheight)
	//	{
	//		cv::Rect rectTmp(y, x, blockwidth, blockheight);
	//		Mat p = chushi_clone2(rectTmp).clone();
	//		pk.push_back(p);
	//		std::vector<double> col;
	//		for (int i = 0; i < p.rows; i++)
	//		{
	//			std::vector<int> row;
	//			for (int j = 0; j < p.cols; j++)
	//			{
	//				int tmp = p.at<int>(i, j);
	//				row.push_back(tmp);
	//			}
	//			int sum = std::accumulate(std::begin(row), std::end(row), 0);
	//			double mean = (double)sum / row.size(); 
	//			col.push_back(mean);
	//		}
	//		for (int j = 0; j < col.size(); j++)
	//		{
	//			Y.at<double>(j, blocknum) = col[j];
	//		}
	//		blocknum++;
	//	}
	//}
	//std::cout << "acutally blocknum:" << blocknum << std::endl;
	//std::cout << "Y.size:" << Y.cols << " " << Y.rows << std::endl;
	//std::cout << "pk.num:" << pk.size() << std::endl;
	//std::cout << "Y.type:" << Y.type() << std::endl;

	////Mat temp;
	//////int multiple = 2;
	////if (Y.cols > Y.rows) 
	////{
	////	int multiple = Y.cols / Y.rows;
	////	cv::resize(Y, temp, cv::Size(Y.cols / multiple, Y.rows), cv::INTER_AREA);
	////}
	////else if (Y.cols <= Y.rows)
	////{
	////	int multiple = Y.rows / Y.cols;
	////	cv::resize(Y, temp, cv::Size(Y.cols, Y.rows / multiple), cv::INTER_AREA);
	////}
	//Mat temp(Y);
	//std::cout << "temp.size:" << temp.cols << " " << temp.rows << std::endl;
	//Mat Y_sparse_lowrank, Y_sparse;
	//low_rank_decomposition(temp, Y_sparse_lowrank, Y_sparse);
	//cv::resize(temp, Y_sparse, Y.size(), cv::INTER_CUBIC);
	//std::cout << "Y_sparse.size:" << Y_sparse.cols << " " << Y_sparse.rows << std::endl;
	//


	//std::vector<double> S;
	//for (int k = 0; k < Y_sparse.cols; k++)
	//{
	//	Mat col_tmp = Mat::zeros(blockheight, 1, CV_64FC1);
	//	for (int t = 0; t < col_tmp.rows; t++)
	//	{
	//		col_tmp.at<double>(t, 0) = (double)Y_sparse.at<double>(t, k);
	//	}
	//	double sp = cv::norm(col_tmp);
	//	S.push_back(sp);
	//}
	//std::cout << "S.size:" << S.size() << std::endl;


	//float d1 = 0.8, d2 = 0.2;
	//float u1 = 0.8, u2 = 0.2;
	//int block = 0;
	//for (int y = 0; y < chushi_clone2.cols - blockwidth + 1; y = y + blockwidth)
	//{
	//	for (int x = 0; x < chushi_clone2.rows - blockheight + 1; x = x + blockheight)
	//	{
	//		cv::Rect rectTmp(y, x, blockwidth, blockheight);
	//		//Mat p = chushi_clone2(rectTmp);
	//		if (S[block] > d2 && S[block] <= d1)
	//		{
	//			chushi_clone2(rectTmp) = u1*pk[block];
	//		}
	//		else if (S[block] <= d2)
	//		{
	//			chushi_clone2(rectTmp) = u2 * pk[block];
	//		}
	//		else
	//		{
	//			continue;
	//		}
	//		block++;
	//	}
	//}
	//Mat jubu = chushi_clone2;
	////imshow("jubu", jubu);

	//float r1 = 0.6, r2 = 0.4;
	//Mat final = quanju * 0.6 + jubu * 0.4;
	//cv::normalize(final, final, 0, 255, cv::NORM_MINMAX);
	//final.convertTo(final, CV_8UC1);
	////imshow("final", final);

	output = chushi.clone();
}