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

void get_sr_saliency(Mat input, Mat &output)
{
	Mat sr_in = input.clone();
	if (sr_in.channels() == 3)
		cvtColor(sr_in, sr_in, cv::COLOR_RGB2GRAY);
	Mat planes[] = { cv::Mat_<float>(sr_in), Mat::zeros(sr_in.size(), CV_32F) };
	Mat complexI; //复数矩阵
	merge(planes, 2, complexI); //把单通道矩阵组合成复数形式的双通道矩阵
	dft(complexI, complexI);  // 使用离散傅立叶变换

	//对复数矩阵进行处理，方法为谱残差
	Mat mag, pha, mag_mean;
	Mat Re, Im;
	split(complexI, planes); //分离复数到实部和虚部
	Re = planes[0]; //实部
	Im = planes[1]; //虚部
	magnitude(Re, Im, mag); //计算幅值
	phase(Re, Im, pha); //计算相角

	float *pre, *pim, *pm, *pp;
	//对幅值进行对数化
	for (int i = 0; i < mag.rows; i++)
	{
		pm = mag.ptr<float>(i);
		for (int j = 0; j < mag.cols; j++)
		{
			*pm = log(*pm);
			pm++;
		}
	}
	blur(mag, mag_mean, cv::Size(5, 5)); //对数谱的均值滤波
	mag = mag - mag_mean; //求取对数频谱残差
	//把对数谱残差的幅值和相角划归到复数形式
	for (int i = 0; i < mag.rows; i++)
	{
		pre = Re.ptr<float>(i);
		pim = Im.ptr<float>(i);
		pm = mag.ptr<float>(i);
		pp = pha.ptr<float>(i);
		for (int j = 0; j < mag.cols; j++)
		{
			*pm = exp(*pm);
			*pre = *pm * cos(*pp);
			*pim = *pm * sin(*pp);
			pre++;
			pim++;
			pm++;
			pp++;
		}
	}
	Mat planes1[] = { cv::Mat_<float>(Re), cv::Mat_<float>(Im) };

	merge(planes1, 2, complexI); //重新整合实部和虚部组成双通道形式的复数矩阵
	idft(complexI, complexI, cv::DFT_SCALE); // 傅立叶反变换
	split(complexI, planes); //分离复数到实部和虚部
	Re = planes[0];
	Im = planes[1];
	magnitude(Re, Im, mag); //计算幅值和相角
	for (int i = 0; i < mag.rows; i++)
	{
		pm = mag.ptr<float>(i);
		for (int j = 0; j < mag.cols; j++)
		{
			*pm = (*pm) * (*pm);
			pm++;
		}
	}
	GaussianBlur(mag, mag, cv::Size(7, 7), 2.5, 2.5);
	Mat invDFT, invDFTcvt;
	normalize(mag, invDFT, 0, 255, cv::NORM_MINMAX); //归一化到[0,255]供显示
	invDFT.convertTo(invDFTcvt, CV_8U); //转化成CV_8U型
	//imshow("SpectualResidual", invDFTcvt);
	//imshow("Original Image", I);
	output = invDFTcvt.clone();
}

void get_bms_saliency(Mat input, Mat &output)
{
	Mat src_small;
	float w = (float)input.cols, h = (float)input.rows;
	float maxD = max(w, h);
	
	resize(input, src_small, cv::Size((int)(400 *w / maxD), (int)(400 *h / maxD)), 0.0, 0.0, cv::INTER_AREA);// standard: width: 600 pixel

	BMS bms(src_small, 3, 1, 0, 1, 1);

	int step = 8;

	bms.computeSaliency((double)step);

	Mat result = bms.getSaliencyMap();

	dilate(result, result, Mat(), cv::Point(-1, -1), 11);

	int blur_width = (int)MIN(floor(20) * 4 + 1, 51);
	GaussianBlur(result, result, cv::Size(blur_width, blur_width), 20, 20);
	
	resize(result, output, input.size());
}

void get_hc_saliency(Mat input, Mat &output)
{
	cv::Mat outputImamge = cv::Mat(input.rows, input.cols, CV_8UC1);
	HC hc;
	hc.calculateSaliencyMap(input, outputImamge);
	output = outputImamge.clone();
}

void get_uav_saliency(Mat input, Mat &output)
{
	int theta = 5;
	calculate_contrast(input, theta, output);
}

void get_myphase_saliency(Mat input, Mat &output)
{
	Mat myphase_in = input.clone();
	//cvtColor(myphase_in, myphase_in, cv::COLOR_RGB2GRAY);
	MyGammaCorrection(myphase_in, myphase_in, 5);
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(myphase_in, grad_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(myphase_in, grad_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, myphase_in);
	Mat tmp;
	get_sr_saliency(myphase_in, tmp);
	output = tmp.clone();
}

void get_saliency_rects(Mat input, int saliency_tpye, vector<cv::Rect> &output_rects)
{
	Mat saliency_in = input.clone();
	Mat saliency;

	if (saliency_tpye == sr_type_)
		get_sr_saliency(saliency_in, saliency);
	else if (saliency_tpye == bms_type_) 
		get_bms_saliency(saliency_in, saliency);
	else if (saliency_tpye == hc_type_)
		get_hc_saliency(saliency_in, saliency);
	else if (saliency_tpye == uav_type_)
		get_uav_saliency(saliency_in, saliency);
	else if (saliency_tpye == myphase_type_)
		get_myphase_saliency(saliency_in, saliency);
	else if (saliency_tpye == myphase2_type_)
		get_myphase_saliency2(saliency_in, saliency);
	else
		cout << "NO THIS SALIENCY TPYE." << endl;

	//cout << saliency.type() << " " << saliency.size() << " " << saliency.channels() << endl;


	if (saliency.channels() != 1)
		cvtColor(saliency, saliency, cv::COLOR_RGB2GRAY);
	if (saliency.type() != CV_8UC1)
		saliency.convertTo(saliency, CV_8UC1);

	Mat bin_sr;
	threshold(saliency, bin_sr, 0, 255, cv::THRESH_OTSU);
	Mat element_sr = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
	morphologyEx(bin_sr, bin_sr, cv::MORPH_CLOSE, element_sr);
	morphologyEx(bin_sr, bin_sr, cv::MORPH_DILATE, element_sr);
	vector<vector<cv::Point> >contours_sr;
	vector<cv::Vec4i> hierarchy_sr;
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
	//Mat tmp = Mat::zeros(bin_sr.size(),CV_8UC3);
	//cv::drawContours(tmp,contours_sr,-1,cv::Scalar(0,255,0));
	//cv::imshow("all_contours",tmp);
	//cv::imwrite("con.jpg",tmp);
}

void low_rank_decomposition(Mat input, Mat &z, Mat &e)
{
	MatrixXd mat(input.rows, input.cols);
	cv::cv2eigen(input, mat);

	double lambda = 0.01;
	LowRankRepresentation mylrr;
	vector<MatrixXd> ZE = mylrr.result(mat, lambda);

	cv::eigen2cv(ZE[0], z);
	cv::eigen2cv(ZE[1], e);
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
	imshow("zengqiang", myphase_in);
	cv::imwrite("zengqiang.jpg", myphase_in);

	Mat real, imag;
	fast_dft(myphase_in, real, imag);
	Mat img_range;
	one_amplitude(real, imag, img_range);      
	Mat tmp = fourior_inverser(img_range);
	Mat chushi = tmp(cv::Rect(0, 0, input.cols, input.rows));
	//imshow("chushi", chushi);


	int desize = 40;
	Mat chushi_clone1 = chushi.clone();
	cv::resize(chushi_clone1, chushi_clone1, cv::Size(desize, desize), cv::INTER_AREA);
	Mat low_rank_part, sparse_part;
	low_rank_decomposition(chushi_clone1, low_rank_part, sparse_part);
	//Mat quanju = chushi_clone1 - low_rank_part;
	Mat quanju = sparse_part.clone();
	cv::resize(quanju, quanju, input.size(), cv::INTER_CUBIC);
	//imshow("quanju", quanju);


	Mat chushi_clone2 = chushi.clone();
	int blocknum = 0;
	int blocksize = 50;
	int num = (chushi_clone2.cols / blocksize)*(chushi_clone2.cols / blocksize);
	Mat Y = Mat::zeros(blocksize, num, CV_64FC1);
	vector<Mat> pk;
	for (int y = 0; y < chushi_clone2.cols - blocksize + 1; y = y + blocksize)
	{
		for (int x = 0; x < chushi_clone2.rows - blocksize + 1; x = x + blocksize)
		{
			cv::Rect rectTmp(y, x, blocksize, blocksize);
			Mat p = chushi_clone2(rectTmp).clone();
			pk.push_back(p);
			vector<double> col;
			for (int i = 0; i < p.rows; i++)
			{
				vector<int> row;
				for (int j = 0; j < p.cols; j++)
				{
					int tmp = p.at<int>(i, j);
					row.push_back(tmp);
				}
				int sum = std::accumulate(std::begin(row), std::end(row), 0);
				double mean = (double)sum / row.size(); 
				col.push_back(mean);
			}

			for (int j = 0; j < col.size(); j++)
			{
				Y.at<double>(j, blocknum) = col[j];
			}

			blocknum++;
		}
	}
	Mat low_rank_part1, sparse_part1;
	low_rank_decomposition(Y, low_rank_part1, sparse_part1);
	//Mat jubu = Y - low_rank_part1;
	Mat jubu = sparse_part1.clone();
	vector<double> S;
	for (int k = 0; k < jubu.cols; k++)
	{
		Mat col_tmp = Mat::zeros(blocksize, 1, CV_64FC1);
		for (int t = 0; t < col_tmp.rows; t++)
		{
			col_tmp.at<double>(t, 0) = (double)jubu.at<double>(t, k);
		}
		double sp = cv::norm(col_tmp);
		S.push_back(sp);
		//cout << sp << endl;
	}
	float d1 = 0.8, d2 = 0.2;
	float u1 = 0.8, u2 = 0.2;
	int block = 0;
	for (int y = 0; y < chushi_clone2.cols - blocksize + 1; y = y + blocksize)
	{
		for (int x = 0; x < chushi_clone2.rows - blocksize + 1; x = x + blocksize)
		{
			cv::Rect rectTmp(y, x, blocksize, blocksize);
			//Mat p = chushi_clone2(rectTmp);
			if (S[block] > d2 && S[block] <= d1)
			{
				chushi_clone2(rectTmp) = u1*pk[block];
			}
			else if (S[block] <= d2)
			{
				chushi_clone2(rectTmp) = u2 * pk[block];
			}
			else
			{
				continue;
			}
			block++;
		}
	}
	jubu = chushi_clone2;
	//imshow("jubu", jubu);


	//float r1 = 0.6, r2 = 0.4;
	//Mat final = quanju * 0.6 + jubu * 0.4;
	

	Mat e1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	Mat e2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

	cv::normalize(chushi, chushi, 0, 255, cv::NORM_MINMAX);
	chushi.convertTo(chushi, CV_8UC1);
	threshold(chushi, chushi, 80, 255, cv::THRESH_BINARY);
	imshow("chushi", chushi);
	cv::imwrite("chushi.jpg", chushi);

	cv::normalize(jubu, jubu, 0, 255, cv::NORM_MINMAX);
	jubu.convertTo(jubu, CV_8UC1);
	threshold(jubu, jubu, 100, 255, cv::THRESH_BINARY);
	//morphologyEx(jubu, jubu, cv::MORPH_ERODE, e1);
	morphologyEx(jubu, jubu, cv::MORPH_DILATE, e2);
	imshow("jubu", jubu);
	cv::imwrite("jubu.jpg", jubu);
	


	cv::normalize(quanju, quanju, 0, 255, cv::NORM_MINMAX);
	quanju.convertTo(quanju, CV_8UC1);
	threshold(quanju, quanju, 100, 255, cv::THRESH_BINARY);
	morphologyEx(quanju, quanju, cv::MORPH_ERODE, e1);
	morphologyEx(quanju, quanju, cv::MORPH_DILATE, e2);
	imshow("quanju", quanju);
	cv::imwrite("quanju.jpg", quanju);

	Mat final = quanju & jubu;

	cv::normalize(final, final, 0, 255, cv::NORM_MINMAX);
	final.convertTo(final, CV_8UC1);
	threshold(final, final, 120, 255, cv::THRESH_BINARY);
	morphologyEx(final, final, cv::MORPH_ERODE, e1);
	morphologyEx(final, final, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
	imshow("final", final);
	cv::imwrite("final.jpg", final);



	output = final.clone();
}