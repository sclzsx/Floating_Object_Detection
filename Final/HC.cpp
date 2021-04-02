#include "HC.h"
using namespace std;
 
template<typename T> inline T sqr(T x) { return x * x; }
typedef std::vector<string> vecS;
typedef std::vector<int> vecI;
typedef std::vector<float> vecF;
typedef std::vector<double> vecD;
typedef pair<double, int> CostIdx;
typedef pair<float, int> CostfIdx;
template<class T> inline T vecDist3(const cv::Vec<T, 3> &v1, const cv::Vec<T, 3> &v2) { return sqrt(sqr(v1[0] - v2[0]) + sqr(v1[1] - v2[1]) + sqr(v1[2] - v2[2])); }
template<class T> inline T vecSqrDist3(const cv::Vec<T, 3> &v1, const cv::Vec<T, 3> &v2) { return sqr(v1[0] - v2[0]) + sqr(v1[1] - v2[1]) + sqr(v1[2] - v2[2]); }

void HC::calculateSaliencyMap(cv::Mat src, cv::Mat &dst)
{
	cv::Mat img3f;
	(src).convertTo(img3f, CV_32FC3, 1.0 / 255);
	cv::Mat idx1i, binColor3f, colorNums1i, weight1f, _colorSal;
	Quantize(img3f, idx1i, binColor3f, colorNums1i);
	cvtColor(binColor3f, binColor3f, CV_BGR2Lab);

	normalize(colorNums1i, weight1f, 1, 0, cv::NORM_L1, CV_32F);
	GetHC(binColor3f, weight1f, _colorSal);
	float* colorSal = (float*)(_colorSal.data);
	cv::Mat salHC1f(img3f.size(), CV_32F);
	for (int r = 0; r < img3f.rows; r++)
	{
		float* salV = salHC1f.ptr<float>(r);
		int* _idx = idx1i.ptr<int>(r);
		for (int c = 0; c < img3f.cols; c++)
			salV[c] = colorSal[_idx[c]];
	}
	GaussianBlur(salHC1f, salHC1f, cv::Size(3, 3), 0);

	normalize(salHC1f, dst, 0, 1, cv::NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1, 255);
}

int HC::Quantize(const cv::Mat& img3f, cv::Mat &idx1i, cv::Mat &_color3f, cv::Mat &_colorNum, double ratio )
{
	static const int clrNums[3] = { 12, 12, 12 };
	static const float clrTmp[3] = { clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f };
	static const int w[3] = { clrNums[1] * clrNums[2], clrNums[2], 1 };

	CV_Assert(img3f.data != NULL);
	idx1i = cv::Mat::zeros(img3f.size(), CV_32S);
	int rows = img3f.rows, cols = img3f.cols;
	if (img3f.isContinuous() && idx1i.isContinuous())
	{
		cols *= rows;
		rows = 1;
	}

	// Build color pallet
	std::map<int, int> pallet;
	for (int y = 0; y < rows; y++)
	{
		const float* imgData = img3f.ptr<float>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++, imgData += 3)
		{
			idx[x] = (int)(imgData[0] * clrTmp[0])*w[0] + (int)(imgData[1] * clrTmp[1])*w[1] + (int)(imgData[2] * clrTmp[2]);
			pallet[idx[x]] ++;
		}
	}

	// Fine significant colors
	int maxNum = 0;
	{
		int count = 0;
		std::vector<pair<int, int>> num; // (num, color) pairs in num
		num.reserve(pallet.size());
		for (map<int, int>::iterator it = pallet.begin(); it != pallet.end(); it++)
			num.push_back(pair<int, int>(it->second, it->first)); // (color, num) pairs in pallet
		sort(num.begin(), num.end(), std::greater< pair<int, int> >());

		maxNum = (int)num.size();
		int maxDropNum = cvRound(rows * cols * (1 - ratio));
		for (int crnt = num[maxNum - 1].first; crnt < maxDropNum && maxNum > 1; maxNum--)
			crnt += num[maxNum - 2].first;
		maxNum = min(maxNum, 256); // To avoid very rarely case
		if (maxNum < 10)
			maxNum = min((int)pallet.size(), 100);
		pallet.clear();
		for (int i = 0; i < maxNum; i++)
			pallet[num[i].second] = i;

		std::vector<cv::Vec3i> color3i(num.size());
		for (unsigned int i = 0; i < num.size(); i++)
		{
			color3i[i][0] = num[i].second / w[0];
			color3i[i][1] = num[i].second % w[0] / w[1];
			color3i[i][2] = num[i].second % w[1];
		}

		for (unsigned int i = maxNum; i < num.size(); i++)
		{
			int simIdx = 0, simVal = INT_MAX;
			for (int j = 0; j < maxNum; j++)
			{
				int d_ij = vecSqrDist3(color3i[i], color3i[j]);
				if (d_ij < simVal)
					simVal = d_ij, simIdx = j;
			}
			pallet[num[i].second] = pallet[num[simIdx].second];
		}
	}
	
	_color3f = cv::Mat::zeros(1, maxNum, CV_32FC3);
	_colorNum = cv::Mat::zeros(_color3f.size(), CV_32S);

	cv::Vec3f* color = (cv::Vec3f*)(_color3f.data);
	int* colorNum = (int*)(_colorNum.data);
	for (int y = 0; y < rows; y++)
	{
		const cv::Vec3f* imgData = img3f.ptr<cv::Vec3f>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++)
		{
			idx[x] = pallet[idx[x]];
			color[idx[x]] += imgData[x];
			colorNum[idx[x]] ++;
		}
	}
	for (int i = 0; i < _color3f.cols; i++)
		color[i] = color[i] / colorNum[i];
	

	return _color3f.cols;
}
void HC::GetHC(const cv::Mat &binColor3f, const cv::Mat &weight1f, cv::Mat &_colorSal)
{
	int binN = binColor3f.cols;
	_colorSal = cv::Mat::zeros(1, binN, CV_32F);
	float* colorSal = (float*)(_colorSal.data);
	std::vector<std::vector< pair<double, int>>> similar(binN); // Similar color: how similar and their index
	cv::Vec3f* color = (cv::Vec3f*)(binColor3f.data);
	float *w = (float*)(weight1f.data);
	for (int i = 0; i < binN; i++)
	{
		std::vector< pair<double, int>> &similari = similar[i];
		similari.push_back(make_pair(0.f, i));
		for (int j = 0; j < binN; j++)
		{
			if (i == j)
				continue;
			float dij = vecDist3<float>(color[i], color[j]);
			similari.push_back(make_pair(dij, j));
			colorSal[i] += w[j] * dij;
		}
		sort(similari.begin(), similari.end());
	}

	SmoothSaliency(binColor3f, _colorSal, 4.0f, similar);
	similar.clear();
	
}
void HC::SmoothSaliency(const cv::Mat &binColor3f, cv::Mat &sal1d, float delta, const std::vector<std::vector< std::pair<double, int> >> &similar)
{
	if (sal1d.cols < 2)
		return;
	CV_Assert(binColor3f.size() == sal1d.size() && sal1d.rows == 1);
	int binN = binColor3f.cols;
	cv::Vec3f* color = (cv::Vec3f*)(binColor3f.data);
	cv::Mat tmpSal;
	sal1d.copyTo(tmpSal);
	float *sal = (float*)(tmpSal.data);
	float *nSal = (float*)(sal1d.data);

	//* Distance based smooth
	int n = max(cvRound(binN / delta), 2);
	vecF dist(n, 0), val(n);
	for (int i = 0; i < binN; i++)
	{
		const std::vector< pair<double, int>> &similari = similar[i];
		float totalDist = 0;

		val[0] = sal[i];
		for (int j = 1; j < n; j++)
		{
			int ithIdx = similari[j].second;
			dist[j] = similari[j].first;
			val[j] = sal[ithIdx];
			totalDist += dist[j];
		}
		float valCrnt = 0;
		for (int j = 0; j < n; j++)
			valCrnt += val[j] * (totalDist - dist[j]);

		nSal[i] = valCrnt / ((n - 1) * totalDist);
	}

}
