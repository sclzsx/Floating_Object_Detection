#include "fft.h"


//�����ȹ�һ����Ǳ��ֲ���
void one_amplitude(Mat complex_r, Mat complex_i, Mat &dst)
{
	Mat temp[] = { Mat::zeros(complex_r.size(),CV_64FC1), Mat::zeros(complex_i.size(),CV_64FC1) };
	float realv = 0.0, imaginv = 0.0;
	for (int i = 0; i < complex_r.cols; i++) {
		for (int j = 0; j < complex_r.rows; j++) {
			realv = complex_r.at<float>(i, j);
			imaginv = complex_i.at<float>(i, j);
			float distance = sqrt(realv*realv + imaginv * imaginv);
			temp[0].at<float>(i, j) = realv / distance;
			temp[1].at<float>(i, j) = imaginv / distance;
		}
	}
	merge(temp, 2, dst);
}

cv::Mat fourior_inverser(Mat &_complexim)
{
	Mat dst;
	Mat iDft[] = { Mat::zeros(_complexim.size(),CV_64FC1),Mat::zeros(_complexim.size(),CV_64FC1) };//��������ͨ��������Ϊfloat����СΪ����ĳߴ�
	idft(_complexim, _complexim);//����Ҷ��任
	split(_complexim, iDft);//���ò��Ҳ�Ǹ���
	magnitude(iDft[0], iDft[1], dst);//����ͨ������Ҫ��ȡ0ͨ��
//    dst += Scalar::all(1);                    // switch to logarithmic scale
//    log(dst, dst);
	//��һ������float���͵���ʾ��ΧΪ0-255,255Ϊ��ɫ��0Ϊ��ɫ
	normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
	return dst;
}

void fast_dft(cv::Mat &src_img, cv::Mat &real_img, cv::Mat &ima_img)
{
	Mat sr_in = src_img.clone();
	if (sr_in.channels() == 3)
		cvtColor(sr_in, sr_in, cv::COLOR_RGB2GRAY);
	Mat planes[] = { cv::Mat_<float>(sr_in), Mat::zeros(sr_in.size(), CV_64FC1) };
	Mat complexI; //��������
	merge(planes, 2, complexI); //�ѵ�ͨ��������ϳɸ�����ʽ��˫ͨ������
	dft(complexI, complexI);  // ʹ����ɢ����Ҷ�任
	split(complexI, planes); //���븴����ʵ�����鲿
	real_img = planes[0]; //ʵ��
	ima_img = planes[1]; //�鲿


	//if (src_img.channels() != 1)
	//	cvtColor(src_img, src_img, cv::COLOR_RGB2GRAY);
	//src_img.convertTo(src_img, CV_64FC1);
	/////////////////////////////////////////���ٸ���Ҷ�任/////////////////////////////////////////////////////
	//int oph = cv::getOptimalDFTSize(src_img.rows);
	//int opw = cv::getOptimalDFTSize(src_img.cols);
	//Mat padded;
	//copyMakeBorder(src_img, padded, 0, oph - src_img.rows, 0, opw - src_img.cols,
	//	cv::BORDER_CONSTANT, cv::Scalar::all(0));
	//Mat temp[] = { padded, Mat::zeros(padded.size(),CV_64FC1) };
	//Mat complexI;
	//merge(temp, 2, complexI);
	//dft(complexI, complexI);    //����Ҷ�任
	//split(complexI, temp);      //��ʾƵ��ͼ
	//temp[0].copyTo(real_img);
	//temp[1].copyTo(ima_img);
}
