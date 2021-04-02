#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include "slic.h"

using namespace cv;
using namespace std;

// For superpixels
const int dx4[4] = { -1, 0, 1, 0 };
const int dy4[4] = { 0, -1, 0, 1 };
//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

// For supervoxels
const int dx10[10] = { -1, 0, 1, 0, -1, 1, 1, -1, 0, 0 };
const int dy10[10] = { 0, -1, 0, 1, -1, -1, 1, 1, 0, 0 };
const int dz10[10] = { 0, 0, 0, 0, 0, 0, 0, 0, -1, 1 };

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

IPSG::SLIC::SLIC()
{
	m_lvec = NULL;
	m_avec = NULL;
	m_bvec = NULL;

	m_lvecvec = NULL;
	m_avecvec = NULL;
	m_bvecvec = NULL;

	bufferGray = NULL;
	bufferRGB = NULL;

	m_nRectX = 0;
	m_nRectY = 0;

}

IPSG::SLIC::~SLIC()
{
	if (m_lvec) delete[] m_lvec;
	if (m_avec) delete[] m_avec;
	if (m_bvec) delete[] m_bvec;


	if (m_lvecvec)
	{
		for (int d = 0; d < m_depth; d++) delete[] m_lvecvec[d];
		delete[] m_lvecvec;
	}
	if (m_avecvec)
	{
		for (int d = 0; d < m_depth; d++) delete[] m_avecvec[d];
		delete[] m_avecvec;
	}
	if (m_bvecvec)
	{
		for (int d = 0; d < m_depth; d++) delete[] m_bvecvec[d];
		delete[] m_bvecvec;
	}

	if (bufferGray) {
		delete[] bufferGray;
	}

	if (bufferRGB) {
		delete[] bufferRGB;
	}

	if (label) {
		delete[] label;
	}
}

//==============================================================================
///	RGB2XYZ
///
/// sRGB (D65 illuminant assumption) to XYZ conversion
//==============================================================================
void IPSG::SLIC::RGB2XYZ(
	const int&		sR,
	const int&		sG,
	const int&		sB,
	double&			X,
	double&			Y,
	double&			Z)
{
	double R = sR / 255.0;
	double G = sG / 255.0;
	double B = sB / 255.0;

	double r, g, b;

	if (R <= 0.04045)	r = R / 12.92;
	else				r = pow((R + 0.055) / 1.055, 2.4);// double pow( double x, double y ); 函数功能: 计算x的y次幂
	if (G <= 0.04045)	g = G / 12.92;
	else				g = pow((G + 0.055) / 1.055, 2.4);
	if (B <= 0.04045)	b = B / 12.92;
	else				b = pow((B + 0.055) / 1.055, 2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
}

//===========================================================================
///	RGB2LAB
//===========================================================================
void IPSG::SLIC::RGB2LAB(const int& sR, const int& sG, const int& sB, double& lval, double& aval, double& bval)
{
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	double X, Y, Z;
	RGB2XYZ(sR, sG, sB, X, Y, Z);

	//------------------------
	// XYZ to LAB conversion
	//------------------------
	double epsilon = 0.008856;	//actual CIE standard
	double kappa = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X / Xr;
	double yr = Y / Yr;
	double zr = Z / Zr;

	double fx, fy, fz;
	if (xr > epsilon)	fx = pow(xr, 1.0 / 3.0);
	else				fx = (kappa*xr + 16.0) / 116.0;
	if (yr > epsilon)	fy = pow(yr, 1.0 / 3.0);
	else				fy = (kappa*yr + 16.0) / 116.0;
	if (zr > epsilon)	fz = pow(zr, 1.0 / 3.0);
	else				fz = (kappa*zr + 16.0) / 116.0;

	lval = 116.0*fy - 16.0;
	aval = 500.0*(fx - fy);
	bval = 200.0*(fy - fz);
}

//===========================================================================
///	DoRGBtoLABConversion
///
///	For whole image: overlaoded floating point version
//===========================================================================
void IPSG::SLIC::DoRGBtoLABConversion( //convert RGB to LAB
	unsigned int*&		ubuff,
	double*&					lvec,
	double*&					avec,
	double*&					bvec)
{
	int sz = m_width*m_height;
	lvec = new double[sz];
	avec = new double[sz];
	bvec = new double[sz];

	for (int j = 0; j < sz; j++)
	{
		int r = (ubuff[j] >> 16) & 0xFF;
		int g = (ubuff[j] >> 8) & 0xFF;
		int b = (ubuff[j]) & 0xFF;

		RGB2LAB(r, g, b, lvec[j], avec[j], bvec[j]);
	}

}

//===========================================================================
///	DoRGBtoLABConversion
///
/// For whole volume
//===========================================================================
void IPSG::SLIC::DoRGBtoLABConversion( //重载转化函数
	unsigned int**&		ubuff,
	double**&					lvec,
	double**&					avec,
	double**&					bvec)
{
	int sz = m_width*m_height;
	for (int d = 0; d < m_depth; d++)
	{
		for (int j = 0; j < sz; j++)
		{
			int r = (ubuff[d][j] >> 16) & 0xFF;
			int g = (ubuff[d][j] >> 8) & 0xFF;
			int b = (ubuff[d][j]) & 0xFF;

			RGB2LAB(r, g, b, lvec[d][j], avec[d][j], bvec[d][j]);
		}
	}
}

//=================================================================================
/// DrawContoursAroundSegments 绘制轮廓周围的轮廓
///
/// Internal contour drawing option exists. One only needs to comment the if内部轮廓绘图选项存在。 只需要评论一下if
/// statement inside the loop that looks at neighbourhood. 在循环内查看邻居的语句。
//=================================================================================
void IPSG::SLIC::DrawContoursAroundSegments(
	std::vector<double>&				kseedsl,
	std::vector<double>&				kseedsa,
	std::vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	unsigned int*			ubuff,
	const int*				labels,
	const int&				width,
	const int&				height,
	const cv::Scalar&		color)
{
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	int sz = width*height;

	vector<bool> istaken(sz, false);

	int mainindex(0);
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			int np(0);
			for (int i = 0; i < 8; i++)
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if ((x >= 0 && x < width) && (y >= 0 && y < height))
				{
					int index = y*width + x;

					if (false == istaken[index])//comment this to obtain internal contours对此进行评论以获取内部轮廓
					{
						if (labels[mainindex] != labels[index])
							np++;
					}
				}
			}
			if (np > 1)//change to 2 or 3 for thinner lines
			{
				ubuff[mainindex] = 0;
				ubuff[mainindex] |= (int)color.val[2] << 16; // r
				ubuff[mainindex] |= (int)color.val[1] << 8; // g
				ubuff[mainindex] |= (int)color.val[0];
				//ubuff[mainindex] |= 255 << 16; // r
				//ubuff[mainindex] |= 0 << 8; // g
				//ubuff[mainindex] |= 0;
				istaken[mainindex] = true;
			}
			mainindex++;
		}
	}
}

void IPSG::SLIC::DrawContoursAroundSegments(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	unsigned char*			ubuff,
	const int*				labels,
	const int&				width,
	const int&				height,
	const cv::Scalar&		color)
{
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	int sz = width*height;

	vector<bool> istaken(sz, false);

	int mainindex(0);
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			int np(0);

			for (int i = 0; i < 8; i++)
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if ((x >= 0 && x < width) && (y >= 0 && y < height))
				{
					int index = y*width + x;
					if (false == istaken[index])//comment this to obtain internal contours对此进行评论以获取内部轮廓
					{
						if (labels[mainindex] != labels[index])
							np++;//如果一个点的周围八个点中有不一样标签值得点那么这点的NP就加1
					}
				}
			}
			if (np > 1)//change to 2 or 3 for thinner lines 更细的线条更改为2或3
			{
				ubuff[mainindex] = (uchar)color.val[0];
				istaken[mainindex] = true;
			}
			mainindex++;
		}
	}
}

//=================================================================================
/// DrawContoursAroundSegmentsTwoColors
///
/// Internal contour drawing option exists. One only needs to comment the if
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void IPSG::SLIC::DrawContoursAroundSegmentsTwoColors(
	unsigned int*			img,
	const int*				labels,
	const int&				width,
	const int&				height)
{
	const int dx[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	int sz = width*height;

	vector<bool> istaken(sz, false);

	vector<int> contourx(sz);
	vector<int> contoury(sz);
	int mainindex(0);
	int cind(0);
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			int np(0);
			for (int i = 0; i < 8; i++)
			{
				int x = k + dx[i];
				int y = j + dy[i];

				if ((x >= 0 && x < width) && (y >= 0 && y < height))
				{
					int index = y*width + x;

					//if( false == istaken[index] )//comment this to obtain internal contours
					{
						if (labels[mainindex] != labels[index]) np++;
					}
				}
			}
			if (np > 1)
			{
				contourx[cind] = k;
				contoury[cind] = j;
				istaken[mainindex] = true;
				//img[mainindex] = color;
				cind++;
			}
			mainindex++;
		}
	}

	int numboundpix = cind;//int(contourx.size());

	for (int j = 0; j < numboundpix; j++)
	{
		int ii = contoury[j] * width + contourx[j];
		img[ii] = 0xffffff;
		//----------------------------------
		// Uncomment this for thicker lines
		//----------------------------------
		for (int n = 0; n < 8; n++)
		{
			int x = contourx[j] + dx[n];
			int y = contoury[j] + dy[n];
			if ((x >= 0 && x < width) && (y >= 0 && y < height))
			{
				int ind = y*width + x;
				if (!istaken[ind]) img[ind] = 0;
			}
		}
	}
}


//==============================================================================
///	DetectLabEdges
//==============================================================================
void IPSG::SLIC::DetectLabEdges(
	const double*				lvec,
	const double*				avec,
	const double*				bvec,
	const int&					width,
	const int&					height,
	vector<double>&				edges)
{
	int sz = width*height;

	edges.resize(sz, 0);//resize()，设置大小（size）;
	for (int j = 1; j < height - 1; j++)
	{

		for (int k = 1; k < width - 1; k++)
		{
			int i = j*width + k;

			double dx = (lvec[i - 1] - lvec[i + 1])*(lvec[i - 1] - lvec[i + 1]) +
				(avec[i - 1] - avec[i + 1])*(avec[i - 1] - avec[i + 1]) +
				(bvec[i - 1] - bvec[i + 1])*(bvec[i - 1] - bvec[i + 1]);

			double dy = (lvec[i - width] - lvec[i + width])*(lvec[i - width] - lvec[i + width]) +
				(avec[i - width] - avec[i + width])*(avec[i - width] - avec[i + width]) +
				(bvec[i - width] - bvec[i + width])*(bvec[i - width] - bvec[i + width]);

			//edges[i] = (sqrt(dx) + sqrt(dy));
			edges[i] = (dx + dy);
		}
	}

}

//===========================================================================
///	PerturbSeeds
//===========================================================================
void IPSG::SLIC::PerturbSeeds(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const vector<double>&		edges)
{
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	int numseeds = kseedsl.size();

	for (int n = 0; n < numseeds; n++)
	{
		int ox = kseedsx[n];//original x
		int oy = kseedsy[n];//original y
		int oind = oy*m_width + ox;

		int storeind = oind;
		for (int i = 0; i < 8; i++)
		{
			int nx = ox + dx8[i];//new x
			int ny = oy + dy8[i];//new y

			if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
			{
				int nind = ny*m_width + nx;
				if (edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if (storeind != oind)
		{
			kseedsx[n] = storeind%m_width;
			kseedsy[n] = storeind / m_width;
			kseedsl[n] = m_lvec[storeind];
			kseedsa[n] = m_avec[storeind];
			kseedsb[n] = m_bvec[storeind];
		}
	}
}


//===========================================================================
///	GetLABXYSeeds_ForGivenStepSize
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void IPSG::SLIC::GetLABXYSeeds_ForGivenStepSize(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const int&					STEP,
	const bool&					perturbseeds,
	const vector<double>&		edgemag)
{
	int numseeds(0);
	int n(0);

	//int xstrips = m_width/STEP;
	//int ystrips = m_height/STEP;
	int xstrips = (0.5 + double(m_width) / double(STEP));
	int ystrips = (0.5 + double(m_height) / double(STEP));

	int xerr = m_width - STEP*xstrips;
	int yerr = m_height - STEP*ystrips;

	double xerrperstrip = double(xerr) / double(xstrips);
	double yerrperstrip = double(yerr) / double(ystrips);

	int xoff = STEP / 2;
	int yoff = STEP / 2;
	//-------------------------
	numseeds = xstrips*ystrips;
	//-------------------------
	kseedsl.resize(numseeds);
	kseedsa.resize(numseeds);
	kseedsb.resize(numseeds);
	kseedsx.resize(numseeds);
	kseedsy.resize(numseeds);

	for (int y = 0; y < ystrips; y++)
	{
		int ye = y*yerrperstrip;
		for (int x = 0; x < xstrips; x++)
		{
			int xe = x*xerrperstrip;
			int i = (y*STEP + yoff + ye)*m_width + (x*STEP + xoff + xe);

			kseedsl[n] = m_lvec[i];
			kseedsa[n] = m_avec[i];
			kseedsb[n] = m_bvec[i];
			kseedsx[n] = (x*STEP + xoff + xe);
			kseedsy[n] = (y*STEP + yoff + ye);
			n++;
		}
	}


	if (perturbseeds)
	{
		PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, edgemag);
	}
}

//===========================================================================
///	GetLABXYSeeds_ForGivenK
///
/// The k seed values are taken as uniform spatial pixel samples.k个种子值被视为统一的空间像素样本。
//===========================================================================
void IPSG::SLIC::GetLABXYSeeds_ForGivenK(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	const int&					K,
	const bool&					perturbseeds,
	const vector<double>&		edgemag)
{
	int sz = m_width*m_height;
	double step = sqrt(double(sz) / double(K));
	int T = step;
	int xoff = step / 2;
	int yoff = step / 2;

	int n(0); int r(0);
	for (int y = 0; y < m_height; y++)
	{
		int Y = y*step + yoff;
		if (Y > m_height - 1) break;

		for (int x = 0; x < m_width; x++)
		{
			//int X = x*step + xoff;//square grid
			int X = x*step + (xoff << (r & 0x1));//hex grid
			if (X > m_width - 1) break;

			int i = Y*m_width + X;

			//_ASSERT(n < K);

			//kseedsl[n] = m_lvec[i];
			//kseedsa[n] = m_avec[i];
			//kseedsb[n] = m_bvec[i];
			//kseedsx[n] = X;
			//kseedsy[n] = Y;
			kseedsl.push_back(m_lvec[i]);
			kseedsa.push_back(m_avec[i]);
			kseedsb.push_back(m_bvec[i]);
			kseedsx.push_back(X);
			kseedsy.push_back(Y);
			n++;
		}
		r++;
	}

	if (perturbseeds)
	{
		PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, edgemag);
	}
}


//===========================================================================
///	PerformSuperpixelSegmentation_VariableSandM
///
///	Magic SLIC - no parameters
///
///	Performs k mean segmentation. It is fast because it looks locally, not
/// over the entire image.
/// This function picks the maximum value of color distance as compact factor
/// M and maximum pixel distance as grid step size S from each cluster (13 April 2011).
/// So no need to input a constant value of M and S. There are two clear
/// advantages:
///
/// [1] The algorithm now better handles both textured and non-textured regions
/// [2] There is not need to set any parameters!!!
///
/// SLICO (or SLIC Zero) dynamically varies only the compactness factor S,
/// not the step size S.
//===========================================================================
void IPSG::SLIC::PerformSuperpixelSegmentation_VariableSandM(
	vector<double>&				kseedsl,
	vector<double>&				kseedsa,
	vector<double>&				kseedsb,
	vector<double>&				kseedsx,
	vector<double>&				kseedsy,
	int*						klabels,
	const int&					STEP,
	const int&					NUMITR
	)
{
	int sz = m_width*m_height;
	const int numk = kseedsl.size();
	//double cumerr(99999.9);
	int numitr(0);

	//----------------
	int offset = STEP;
	if (STEP < 10) offset = STEP*1.5;
	//----------------

	vector<double> sigmal(numk, 0);
	vector<double> sigmaa(numk, 0);
	vector<double> sigmab(numk, 0);
	vector<double> sigmax(numk, 0);
	vector<double> sigmay(numk, 0);
	vector<int> clustersize(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values
	vector<double> distxy(sz, DBL_MAX);//DBL_MAX=1.7976931348623158e+308 
	vector<double> distlab(sz, DBL_MAX);
	vector<double> distvec(sz, DBL_MAX);
	vector<double> maxlab(numk, 10 * 10);//THIS IS THE VARIABLE VALUE OF M, just start with 10
	vector<double> maxxy(numk, STEP*STEP);//THIS IS THE VARIABLE VALUE OF M, just start with 10

	double invxywt = 1.0 / (STEP*STEP);//NOTE: this is different from how usual SLIC/LKM works

	while (numitr < NUMITR)
	{
		//------
		//cumerr = 0;
		numitr++;
		//------

		distvec.assign(sz, DBL_MAX);
		for (int n = 0; n < numk; n++)
		{
			int y1 = std::max(0, (int)(kseedsy[n] - offset));
			int y2 = std::min(m_height, (int)(kseedsy[n] + offset));
			int x1 = std::max(0, (int)(kseedsx[n] - offset));
			int x2 = std::min(m_width, (int)(kseedsx[n] + offset));

			for (int y = y1; y < y2; y++)
			{
				for (int x = x1; x < x2; x++)
				{
					int i = y*m_width + x;
					_ASSERT(y < m_height && x < m_width && y >= 0 && x >= 0);

					double l = m_lvec[i];
					double a = m_avec[i];
					double b = m_bvec[i];

					distlab[i] = (l - kseedsl[n])*(l - kseedsl[n]) +
						(a - kseedsa[n])*(a - kseedsa[n]) +
						(b - kseedsb[n])*(b - kseedsb[n]);

					distxy[i] = (x - kseedsx[n])*(x - kseedsx[n]) +
						(y - kseedsy[n])*(y - kseedsy[n]);

					//------------------------------------------------------------------------
					double dist = distlab[i] / maxlab[n] + distxy[i] * invxywt;//only varying m, prettier superpixels
					//double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
					//------------------------------------------------------------------------

					if (dist < distvec[i])
					{
						distvec[i] = dist;
						klabels[i] = n;
					}
				}
			}
		}
		//-----------------------------------------------------------------
		// Assign the max color distance for a cluster
		//-----------------------------------------------------------------
		if (0 == numitr)
		{
			maxlab.assign(numk, 1);
			maxxy.assign(numk, 1);
		}
		{for (int i = 0; i < sz; i++)
		{
			if (maxlab[klabels[i]] < distlab[i]) maxlab[klabels[i]] = distlab[i];
			if (maxxy[klabels[i]] < distxy[i]) maxxy[klabels[i]] = distxy[i];
		}}
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		sigmal.assign(numk, 0);
		sigmaa.assign(numk, 0);
		sigmab.assign(numk, 0);
		sigmax.assign(numk, 0);
		sigmay.assign(numk, 0);
		clustersize.assign(numk, 0);
		//------------------------此函数可以作为显示最终图像使用------------------------------------------------------------
		for (int j = 0; j < sz; j++)
		{
			int temp = klabels[j];
			_ASSERT(klabels[j] >= 0);
			sigmal[klabels[j]] += m_lvec[j];
			sigmaa[klabels[j]] += m_avec[j];
			sigmab[klabels[j]] += m_bvec[j];
			sigmax[klabels[j]] += (j%m_width);
			sigmay[klabels[j]] += (j / m_width);

			clustersize[klabels[j]]++;
		}
		//------------------------此函数可以作为显示最终图像使用------------------------------------------------------------
		{for (int k = 0; k < numk; k++)
		{
			//_ASSERT(clustersize[k] > 0);
			if (clustersize[k] <= 0) clustersize[k] = 1;
			inv[k] = 1.0 / double(clustersize[k]);//computing inverse now to multiply, than divide later
		}}

		{for (int k = 0; k < numk; k++)
		{
			kseedsl[k] = sigmal[k] * inv[k];
			kseedsa[k] = sigmaa[k] * inv[k];
			kseedsb[k] = sigmab[k] * inv[k];
			kseedsx[k] = sigmax[k] * inv[k];
			kseedsy[k] = sigmay[k] * inv[k];
		}}
	}
}

//===========================================================================
///	SaveSuperpixelLabels
///
///	Save labels in raster scan order.
//===========================================================================
void IPSG::SLIC::SaveSuperpixelLabels(
	const int*					labels,
	const int&					width,
	const int&					height,
	const string&				filename,
	const string&				path)
{
	int sz = width*height;

	char fname[_MAX_FNAME];
	char extn[_MAX_FNAME];
	_splitpath(filename.c_str(), NULL, NULL, fname, extn);
	string temp = fname;

	ofstream outfile;
	string finalpath = path + temp + string(".dat");
	outfile.open(finalpath.c_str(), ios::binary);
	for (int i = 0; i < sz; i++)
	{
		outfile.write((const char*)&labels[i], sizeof(int));
	}
	outfile.close();
}

//===========================================================================
///	EnforceLabelConnectivity 强制标签连接
///
///		1. finding an adjacent label for each new component at the start 在开始时为每个新组件找到相邻的标签
///		2. if a certain component is too small, assigning the previously found 如果某个组件太小，则分配先前找到的
///		    adjacent label to this component, and not incrementing the label.与此组件相邻的标签，而不是递增标签。
//=========================================================================== 
void IPSG::SLIC::EnforceLabelConnectivity(
	vector<double>& nlabelcount,
	const int*					labels,//input labels that need to be corrected to remove stray labels 输入标签，需要更正以消除杂散标签
	const int&					width,
	const int&					height,
	int*						nlabels,//new labels新的标签
	int&						numlabels,//the number of labels changes in the end if segments are removed 如果分段被删除，标签的数量将会改变
	const int&					K) //the number of superpixels desired by the user
{
	//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };

	const int sz = width*height;
	nlabelcount.resize(sz);
	const int SUPSZ = sz / K;
	//nlabels.resize(sz, -1);
	for (int i = 0; i < sz; i++)
		nlabels[i] = -1;
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			if (0 > nlabels[oindex])
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed 如果需要，快速找到相邻的标签供以后使用
				//-------------------------------------------------------
				for (int n = 0; n < 4; n++)
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if ((x >= 0 && x < width) && (y >= 0 && y < height))
					{
						int nindex = y*width + x;
						if (nlabels[nindex] >= 0)
							adjlabel = nlabels[nindex];//找一个点上下左右点的值，如果是大于等于0的，就把它交给adjlabel邻居
					}
				}

				int count(1);
				for (int c = 0; c < count; c++)//C表示中心点一直向左移动直到无法计数就停止一行进行编号
				{
					for (int n = 0; n < 4; n++)
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if ((x >= 0 && x < width) && (y >= 0 && y < height))
						{
							int nindex = y*width + x;

							if (0 > nlabels[nindex] && labels[oindex] == labels[nindex])//如果它周边全是小于0的值并且编号的值与周围的值也相等，说明这个点没有邻居的周边
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								nlabelcount[nindex] = (label);

								count++;
							}
						}
					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an 
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------如果分段大小小于极限值，则分配之前找到的相邻标签，并减少标签计数。 


				if (count <= SUPSZ >> 2)
				{
					for (int c = 0; c < count; c++)
					{
						int ind = yvec[c] * width + xvec[c];
						nlabels[ind] = adjlabel;

					}
					label--;
				}
				label++;//这个是用来统计不同编号超像素的个数（对于原始图片有1008个超像素)

			}
			oindex++;//横排扫描发现不同编号超像素的点(即发现-1时开始新的编号)

		}

	}
	numlabels = label;

	if (xvec) delete[] xvec;
	if (yvec) delete[] yvec;

}

//===========================================================================
///	PerformSLICO_ForGivenStepSize
///
/// There is option to save the labels if needed.
//===========================================================================
void IPSG::SLIC::PerformSLICO_ForGivenStepSize(
	unsigned int*			ubuff,
	const int					width,
	const int					height,
	int*						klabels,
	int&						numlabels,
	const int&					STEP,
	const double&				m)
{
	vector<double> kseedsl(0);
	vector<double> kseedsa(0);
	vector<double> kseedsb(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);
	vector<double> nlabelcount(0);
	//--------------------------------------------------
	m_width = width;
	m_height = height;
	int sz = m_width*m_height;
	//klabels.resize( sz, -1 );
	//--------------------------------------------------
	//klabels = new int[sz];
	for (int s = 0; s < sz; s++) klabels[s] = -1;
	//--------------------------------------------------
	DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
	//--------------------------------------------------

	bool perturbseeds(true);
	vector<double> edgemag(0);
	if (perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenStepSize(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, STEP, perturbseeds, edgemag);

	PerformSuperpixelSegmentation_VariableSandM(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, 10);
	numlabels = kseedsl.size();

	int* nlabels = new int[sz];
	EnforceLabelConnectivity(nlabelcount, klabels, m_width, m_height, nlabels, numlabels, double(sz) / double(STEP*STEP));
	{for (int i = 0; i < sz; i++) klabels[i] = nlabels[i]; }
	if (nlabels) delete[] nlabels;
}

//===========================================================================
///	PerformSLICO_ForGivenK
///
/// Zero parameter SLIC algorithm for a given number K of superpixels.
//===========================================================================
void IPSG::SLIC::PerformSLICO_ForGivenK(
	unsigned int*			ubuff,
	const int					width,
	const int					height,
	int*						klabels,
	int&						numlabels,
	const int&					K,//required number of superpixels
	const double&				m,
	cv::Scalar color
	)//weight given to spatial distance[1~40]
{
	vector<double> kseedsl(0);
	vector<double> kseedsa(0);
	vector<double> kseedsb(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);
	vector<double> nlabelcount(0);

	//--------------------------------------------------
	m_width = width;
	m_height = height;
	int sz = m_width*m_height;
	//--------------------------------------------------
	//if(0 == klabels) klabels = new int[sz];
	for (int s = 0; s < sz; s++)
	{

		klabels[s] = -1;

	}
	//--------------------------------------------------
	if (0)//LAB
	{
		DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
	}
	else//RGB
	{
		m_lvec = new double[sz]; m_avec = new double[sz]; m_bvec = new double[sz];
		for (int i = 0; i < sz; i++)
		{
			m_lvec[i] = ubuff[i] >> 16 & 0xff;
			m_avec[i] = ubuff[i] >> 8 & 0xff;
			m_bvec[i] = ubuff[i] & 0xff;
		}
	}
	//--------------------------------------------------

	bool perturbseeds(true);
	vector<double> edgemag(0);
	if (perturbseeds)
		DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenK(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, K, perturbseeds, edgemag);

	int STEP = sqrt(double(sz) / double(K)) + 2.0;//adding a small value in the even the STEP size is too small.
	//PerformSuperpixelSLIC(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, edgemag, m);
	PerformSuperpixelSegmentation_VariableSandM(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, 10);
	numlabels = kseedsl.size();

	int* nlabels = new int[sz];
	std::vector<double> nlabel;
	EnforceLabelConnectivity(nlabelcount, klabels, m_width, m_height, nlabels, numlabels, K);	//	从新编码超像素去掉孤立点
	/*{for (int i = 0; i < sz; i++)
	klabels[i] = nlabels[i]; }
	if (nlabels) delete[] nlabels;*/

	vector<int> labelnum(numlabels, 0);
	labelnum.assign(numlabels, 0);

	{for (int i = 0; i < sz; i++)
	{
		klabels[i] = nlabels[i];
		labelnum[nlabels[i]]++;


	}
	}

	{for (int i = 0; i < sz; i++)
	{
		klabels[i] = nlabels[i];
		labelnum[nlabels[i]]++;


	}
	}
	//int thesumoflabels = labelnum.size();
	//vector<double>::const_iterator s;
	//FILE *fp;
	//fp = fopen("编号代号.txt", "w");
	//ofstream OutFile("编号代号.xlsx", ios::out | ios::binary);
	//int count(0);
	//for (s = nlabelcount.begin(); s != nlabelcount.end(); s++)
	//{
	//	count++;
	//	OutFile << *s <<' ';
	//	if (count == 879)
	//	{
	//		OutFile << endl;
	//		count = 0;
	//	}
	//	
	//}
	//fclose(fp);
	//
	//unsigned char	nlabelcountUINT;
	//unsigned int*nlabelcountUINT = new UINT[sz];
	//cv::Mat labelsmap(m_height, m_width, CV_8UC4);
	//for (int i = 0; i < sz; i++)
	//{
	//	nlabelcountUINT[i] = 0;
	//	nlabelcountUINT[i] |= (int)nlabelcount[i] << 16;  // r
	//	nlabelcountUINT[i] |= (int)nlabelcount[i] << 8; // g
	//	nlabelcountUINT[i] |= (int)nlabelcount[i];
	//	
	//}
	//memcpy(labelsmap.data, nlabelcountUINT, m_width*m_height * sizeof(UINT));
	//cvtColor(labelsmap, labelsmap, CV_BGRA2BGR);


	//nlabelcountUINT[i] = 0;
	//ubuff[mainindex] |= (int)color.val[2] << 16; // r
	//ubuff[mainindex] |= (int)color.val[1] << 8; // g
	//ubuff[mainindex] |= (int)color.val[0];


	//char* nlabelcountUINT = new char[sz];
	//for (int i = 0; i < sz; i++)
	//{
	//	nlabelcountUINT[i]=nlabelcount[i];
	//	
	//}
	//cv::Mat labelsmap(m_height, m_width, CV_8UC1);
	//memcpy(labelsmap.data, nlabelcountUINT, m_width*m_height * sizeof(uchar));

	Mergearea2(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, nlabelcount, ubuff, klabels, m_width, m_height, cv::Scalar(255, 255, 255));

	kseedsl.clear();
	kseedsa.clear();
	kseedsb.clear();
	kseedsx.clear();
	kseedsy.clear();
	nlabelcount.clear();

	if (nlabels) delete[] nlabels;

}

void IPSG::SLIC::PerformSLICO_ForGivenK2(
	unsigned int*			ubuff,
	const int					width,
	const int					height,
	int*						klabels,
	int&						numlabels,
	const int&					K,//required number of superpixels
	const double&				m,
	cv::Scalar color
	)//weight given to spatial distance[1~40]
{
	vector<double> kseedsl(0);
	vector<double> kseedsa(0);
	vector<double> kseedsb(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);
	vector<double> nlabelcount(0);

	//--------------------------------------------------
	m_width = width;
	m_height = height;
	int sz = m_width*m_height;
	//--------------------------------------------------
	//if(0 == klabels) klabels = new int[sz];
	for (int s = 0; s < sz; s++)
	{

		klabels[s] = -1;

	}
	//--------------------------------------------------
	if (0)//LAB
	{
		DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
	}
	else//RGB
	{
		m_lvec = new double[sz]; m_avec = new double[sz]; m_bvec = new double[sz];
		for (int i = 0; i < sz; i++)
		{
			m_lvec[i] = ubuff[i] >> 16 & 0xff;
			m_avec[i] = ubuff[i] >> 8 & 0xff;
			m_bvec[i] = ubuff[i] & 0xff;
		}
	}
	//--------------------------------------------------

	bool perturbseeds(true);
	vector<double> edgemag(0);
	if (perturbseeds)
		DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenK(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, K, perturbseeds, edgemag);

	int STEP = sqrt(double(sz) / double(K)) + 2.0;//adding a small value in the even the STEP size is too small.
	//PerformSuperpixelSLIC(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, edgemag, m);
	PerformSuperpixelSegmentation_VariableSandM(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, 10);
	numlabels = kseedsl.size();

	int* nlabels = new int[sz];
	std::vector<double> nlabel;
	EnforceLabelConnectivity(nlabelcount, klabels, m_width, m_height, nlabels, numlabels, K);	//	从新编码超像素去掉孤立点
	/*{for (int i = 0; i < sz; i++)
	klabels[i] = nlabels[i]; }
	if (nlabels) delete[] nlabels;*/

	vector<int> labelnum(numlabels, 0);
	labelnum.assign(numlabels, 0);

	{for (int i = 0; i < sz; i++)
	{
		klabels[i] = nlabels[i];
		labelnum[nlabels[i]]++;


	}
	}

	{for (int i = 0; i < sz; i++)
	{
		klabels[i] = nlabels[i];
		labelnum[nlabels[i]]++;


	}
	}
	//int thesumoflabels = labelnum.size();
	//vector<double>::const_iterator s;
	//FILE *fp;
	//fp = fopen("编号代号.txt", "w");
	//ofstream OutFile("编号代号.xlsx", ios::out | ios::binary);
	//int count(0);
	//for (s = nlabelcount.begin(); s != nlabelcount.end(); s++)
	//{
	//	count++;
	//	OutFile << *s <<' ';
	//	if (count == 879)
	//	{
	//		OutFile << endl;
	//		count = 0;
	//	}
	//	
	//}
	//fclose(fp);
	//
	//unsigned char	nlabelcountUINT;
	//unsigned int*nlabelcountUINT = new UINT[sz];
	//cv::Mat labelsmap(m_height, m_width, CV_8UC4);
	//for (int i = 0; i < sz; i++)
	//{
	//	nlabelcountUINT[i] = 0;
	//	nlabelcountUINT[i] |= (int)nlabelcount[i] << 16;  // r
	//	nlabelcountUINT[i] |= (int)nlabelcount[i] << 8; // g
	//	nlabelcountUINT[i] |= (int)nlabelcount[i];
	//	
	//}
	//memcpy(labelsmap.data, nlabelcountUINT, m_width*m_height * sizeof(UINT));
	//cvtColor(labelsmap, labelsmap, CV_BGRA2BGR);


	//nlabelcountUINT[i] = 0;
	//ubuff[mainindex] |= (int)color.val[2] << 16; // r
	//ubuff[mainindex] |= (int)color.val[1] << 8; // g
	//ubuff[mainindex] |= (int)color.val[0];


	//char* nlabelcountUINT = new char[sz];
	//for (int i = 0; i < sz; i++)
	//{
	//	nlabelcountUINT[i]=nlabelcount[i];
	//	
	//}
	//cv::Mat labelsmap(m_height, m_width, CV_8UC1);
	//memcpy(labelsmap.data, nlabelcountUINT, m_width*m_height * sizeof(uchar));

	Mergearea2(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, nlabelcount, ubuff, klabels, m_width, m_height, cv::Scalar(255, 255, 255));

	kseedsl.clear();
	kseedsa.clear();
	kseedsb.clear();
	kseedsx.clear();
	kseedsy.clear();
	nlabelcount.clear();

	if (nlabels) delete[] nlabels;

}

void IPSG::SLIC::PerformSLICO_ForGivenK(
	unsigned char*		ubuff,
	const int					width,
	const int					height,
	int*						klabels,
	int&						numlabels,
	const int&					K,//required number of superpixels
	const double&				m

	)//weight given to spatial distance
{
	vector<double> kseedsl(0);
	vector<double> kseedsa(0);
	vector<double> kseedsb(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);
	vector<double> nlabelcount(0);
	//--------------------------------------------------
	m_width = width;
	m_height = height;
	int sz = m_width*m_height;
	//--------------------------------------------------
	//if(0 == klabels) klabels = new int[sz];
	for (int s = 0; s < sz; s++) klabels[s] = -1;
	//--------------------------------------------------

	m_lvec = new double[sz]; m_avec = new double[sz]; m_bvec = new double[sz];
	for (int i = 0; i < sz; i++)
	{
		m_lvec[i] = ubuff[i];
		m_avec[i] = 0;
		m_bvec[i] = 0;
	}


	//--------------------------------------------------

	bool perturbseeds(true);
	vector<double> edgemag(0);
	if (perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenK(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, K, perturbseeds, edgemag);

	int STEP = sqrt(double(sz) / double(K)) + 2.0;//adding a small value in the even the STEP size is too small.
	//PerformSuperpixelSLIC(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, edgemag, m);
	PerformSuperpixelSegmentation_VariableSandM(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, 10);
	numlabels = kseedsl.size();

	int* nlabels = new int[sz];
	EnforceLabelConnectivity(nlabelcount, klabels, m_width, m_height, nlabels, numlabels, K);
	{for (int i = 0; i < sz; i++) klabels[i] = nlabels[i]; }

	if (nlabels) delete[] nlabels;
}

void IPSG::SLIC::GenerateSuperpixels(cv::Mat& img, UINT numSuperpixels)
{
	if (img.empty()) {
		exit(-1);
	}

	int height = img.rows;
	int width = img.cols;
	int sz = height * width;
	label = new int[sz];
	if (img.channels() == 1) {
		type = GRAY;
	}
	else if (img.channels() == 3) {
		type = RGB;
	}
	if (type == GRAY) {
		Mat2Buffer(img, bufferGray);
		PerformSLICO_ForGivenK(bufferGray, img.cols, img.rows, label, sz, numSuperpixels, 10);
	}
	else if (type == RGB) {
		Mat2Buffer(img, bufferRGB);
		PerformSLICO_ForGivenK(bufferRGB, img.cols, img.rows, label, sz, numSuperpixels, 10, cv::Scalar(255, 255, 255));
	}
}

void IPSG::SLIC::GenerateSuperpixels2(cv::Mat& img, UINT numSuperpixels)
{
	if (img.empty()) {
		exit(-1);
	}

	int height = img.rows;
	int width = img.cols;
	int sz = height * width;
	label = new int[sz];
	if (img.channels() == 1) {
		type = GRAY;
	}
	else if (img.channels() == 3) {
		type = RGB;
	}
	if (type == GRAY) {
		Mat2Buffer(img, bufferGray);
		PerformSLICO_ForGivenK(bufferGray, img.cols, img.rows, label, sz, numSuperpixels, 10);
	}
	else if (type == RGB) {
		Mat2Buffer(img, bufferRGB);
		PerformSLICO_ForGivenK2(bufferRGB, img.cols, img.rows, label, sz, numSuperpixels, 10, cv::Scalar(255, 255, 255));
	}
}

// 
int* IPSG::SLIC::GetLabel()
{
	return label;
}

cv::Mat IPSG::SLIC::GetImgWithContours(
	const int&					K,
	cv::Scalar color)
{
	vector<double> kseedsl(0);
	vector<double> kseedsa(0);
	vector<double> kseedsb(0);
	vector<double> kseedsx(0);
	vector<double> kseedsy(0);
	vector<double> nlabelcount(0);
	bool perturbseeds(true);
	vector<double> edgemag(0);
	int sz = m_width*m_height;
	double step = sqrt(double(sz) / double(K));
	int T = step;
	int xoff = step / 2;
	int yoff = step / 2;

	int n(0); int r(0);
	for (int y = 0; y < m_height; y++)
	{
		int Y = y*step + yoff;
		if (Y > m_height - 1) break;

		for (int x = 0; x < m_width; x++)
		{
			//int X = x*step + xoff;//square grid
			int X = x*step + (xoff << (r & 0x1));//hex grid
			if (X > m_width - 1) break;

			int i = Y*m_width + X;

			kseedsl.push_back(m_lvec[i]);
			kseedsa.push_back(m_avec[i]);
			kseedsb.push_back(m_bvec[i]);
			kseedsx.push_back(X);
			kseedsy.push_back(Y);
			n++;
		}
		r++;
	}
	//if (perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	//GetLABXYSeeds_ForGivenK(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, K, perturbseeds, edgemag);


	if (type == GRAY) {
		DrawContoursAroundSegments(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, bufferGray, label, m_width, m_height, color);
		cv::Mat result(m_height, m_width, CV_8UC1);
		memcpy(result.data, bufferGray, m_width*m_height * sizeof(uchar));
		return result;
	}


	else if (type == RGB) {
		DrawContoursAroundSegments(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, bufferRGB, label, m_width, m_height, color);
		cv::Mat result(m_height, m_width, CV_8UC4);
		memcpy(result.data, bufferRGB, m_width*m_height * sizeof(UINT));
		cvtColor(result, result, cv::COLOR_BGRA2BGR);
		return result;
	}

}

void IPSG::SLIC::Mat2Buffer(const cv::Mat& img, UINT*& buffer)
{
	int sz = img.cols * img.rows;
	if (bufferRGB) {
		delete[] bufferRGB;
	}
	bufferRGB = new UINT[sz];

	cv::Mat newImage;
	cv::cvtColor(img, newImage, cv::COLOR_BGR2BGRA);
	memcpy(bufferRGB, (UINT*)newImage.data, sz * sizeof(UINT));
	//memcpy指的是c和c++使用的内存拷贝函数，memcpy函数的功能是从源src所指的内存地址的起始位置开始拷贝n个字节到目标dest所指的内存地址的起始位置中。
}

void IPSG::SLIC::Mat2Buffer(const cv::Mat& img, uchar*& buffer)
{
	int sz = img.cols * img.rows;
	if (bufferGray) {
		delete[] bufferGray;
	}
	bufferGray = new uchar[sz];

	memcpy(bufferGray, (UINT*)img.data, sz * sizeof(uchar));

}

//
//bool IPSG::SLIC::Mergearea(
//	std::vector<double>&        kseedsl,
//	std::vector<double>&        kseedsa,
//	std::vector<double>&        kseedsb,
//	std::vector<double>&        kseedsx,
//	std::vector<double>&        kseedsy,
//	std::vector<double>&        nlabel,
//	unsigned int*		    	ubuff,
//	int*						labels,
//	const int&					width,
//	const int&					height,
//	cv::Scalar color
//)
//
//{
//	int clorfazhi(45), piontnumber(5);
//	double ab, distance;
//	double colora, colorb, colordistancel, colordistancea, colordistanceb;
//	int zdnum, seednum, labVALUE, labVALUE1, labVALUE2, templab;
//	int count(0), colorcount1(0), colorcount2(0), colorcount3(0), indexi, indexj;
//	zdnum = width*height;
//	seednum = kseedsx.size();
//	int* nlabels = new int[zdnum];
//	//int* nlabelcount = new int[zdnum];
//	std::vector<double>        nlabelcount, nlabelcount1, newnlabelcount1, newnlabelcount2, nlabelscolorsite;
//	std::vector<double> clustersize1, numnewnlabelclustersize;
//	nlabelcount1.resize(0);
//	newnlabelcount1.resize(0);
//	newnlabelcount2.resize(0);
//	nlabelscolorsite.resize(0);
//	clustersize1.resize(0);
//	numnewnlabelclustersize.resize(0);
//	nlabelcount.resize(0);
//	for (int i = 0; i < zdnum; i++)
//	{
//		nlabels[i] = labels[i];
//		nlabelcount1.push_back(labels[i]);
//	}
//	bool state(true), state1(true), state2(false), state3(false);
//	int numcountlabel1(2), numcountlabel(0);
//	for (int c = 0; c < 50; c++)
//	{
//		nlabelcount.clear();
//		nlabelcount.push_back(0);
//		for (int i = 0; i < seednum; i++)
//		{
//			colorcount2 = 0;
//			for (int j = 0; j < seednum; j++)
//			{
//				if (i != j)
//				{
//					ab = ((kseedsx[i] - kseedsx[j])* (kseedsx[i] - kseedsx[j]) + (kseedsy[i] - kseedsy[j])*(kseedsy[i] - kseedsy[j]));
//					distance = sqrt(ab);
//					if (distance <40)
//					{
//						indexi = kseedsy[i] * width + kseedsx[i];
//						indexj = kseedsy[j] * width + kseedsx[j];
//						colordistancel = (kseedsl[i] - kseedsl[j]);
//						colordistancea = (kseedsa[i] - kseedsa[j]);
//						colordistanceb = (kseedsb[i] - kseedsb[j]);
//
//						if (((colordistancel >clorfazhi) && (colordistancea > clorfazhi) && (colordistanceb && clorfazhi)) || (colordistancel>125) || (colordistancea > 125) || (colordistanceb > 125))//35
//						{
//							colorcount2++;
//							if (colorcount2 >= piontnumber)//对于40时候可以取5个
//							{
//								int dex(0), num1(0), num2(0);
//								dex = nlabelcount.size();
//
//								num1 = nlabelcount[dex - 1];
//								num2 = nlabelcount1[indexi];
//								if ((num1 < num2) && (num1 < i))
//								{
//									colorcount3++;
//									nlabelcount.push_back(i);
//								}
//							}
//						}
//						else
//							continue;
//					}
//					else
//						continue;
//				}
//				else
//					continue;
//			}
//		}
//		numcountlabel = nlabelcount.size();
//		if (numcountlabel >= 8)
//		{
//			if (clorfazhi < 250)//150
//			{
//				clorfazhi += 5;
//				if (clorfazhi > 75 && piontnumber <= 10)
//				{
//					clorfazhi -= 5;
//					piontnumber += 1;
//				}
//				else
//					continue;
//			}
//			else
//				break;
//		}
//		else if (numcountlabel <= 1)
//		{
//			if (clorfazhi > 45)
//			{
//				clorfazhi -= 5;
//				if (clorfazhi < 5 && piontnumber >= 4)
//				{
//					clorfazhi += 5;
//					piontnumber -= 1;
//				}
//				else
//					continue;
//			}
//			else
//				break;
//		}
//		else
//			break;
//	}
//
//	//int numcountlabel(0);
//	int sz = width*height;
//	int cn1(0), cn2(0);
//
//	//	int numcountlabel = nlabelcount.size();
//	//-------将nlabelcount1重新编码存入	newnlabelcount1中
//
//	int numseeds = kseedsl.size();
//	newnlabelcount1.assign(numseeds, -1);
//	for (int n = 0; n < numseeds; n++)
//	{
//		int ox = kseedsx[n];//original x
//		int oy = kseedsy[n];//original y
//		int oind = oy*m_width + ox;
//
//		int storeind = labels[oind];
//		newnlabelcount1[n] = storeind;
//	}
//
//	for (int k = 1; k < numcountlabel; k++)
//	{
//		newnlabelcount2.push_back(newnlabelcount1[nlabelcount[k]]);
//	}
//
//	int numnewnlabelcount2 = newnlabelcount2.size();
//
//	//---------------------计算选出来的区域大小(即相应编号对应的像素个数)
//
//	clustersize1.assign(seednum + 20, -1);
//	for (int j = 0; j < zdnum; j++)
//	{
//		int temp = nlabelcount1[j];
//		if (nlabelcount1[j] >= 0)
//			clustersize1[nlabelcount1[j]]++;
//	}
//
//	for (int i = 0; i < numnewnlabelcount2; i++)
//	{
//		int temp1 = newnlabelcount2[i];
//		for (int j = 0; j < numseeds; j++)
//		{
//			int temp2 = newnlabelcount1[j];
//			if (temp2 == temp1)
//			{
//				numnewnlabelclustersize.push_back(j);
//
//			}
//			else
//				continue;
//		}
//	}
//
//	//------------------将便利找点并将颜色设置成相应的值并把相关区域显示出来-----
//
//
//	for (int k = 0; k < numnewnlabelcount2; k++)
//	{
//		int temp1 = clustersize1[numnewnlabelclustersize[k]];
//		if (temp1 > 100)//200
//		{
//			int temp = newnlabelcount2[k];
//
//			for (int i = 0; i < zdnum; i++)
//			{
//
//				if (temp == nlabelcount1[i])
//				{
//					nlabelscolorsite.push_back(i);
//					ubuff[i] = (uchar)color.val[1];
//				}
//				else
//					continue;
//			}
//		}
//	}
//
//	//std::cout << "当前图像存在" << numcountlabel -1<< "个目标物体" << std::endl;
//	for (int n = 1; n < numcountlabel; n++)
//	{
//		int  testresultlabelvalue = nlabelcount[n];
//		
//		
//		int ox = kseedsx[testresultlabelvalue];//original x
//		int oy = kseedsy[testresultlabelvalue];//original y
//		
//		if(ox < 0 || oy < 0)
//		{
//			continue;
//		}
//		
//		DrawScalesRects(cv::Point(ox, oy));
//	}
//	//nlabelcount1.clear();
//	//newnlabelcount1.clear();
//	//newnlabelcount2.clear();
//	//nlabelscolorsite.clear();
//	//clustersize1.clear();
//	//numnewnlabelclustersize.clear();
//	//nlabelcount.clear();
//	if (nlabels)
//		delete[] nlabels;
//	return true;
//}

void IPSG::SLIC::Mergearea2(
	std::vector<double>&        kseedsl,
	std::vector<double>&        kseedsa,
	std::vector<double>&        kseedsb,
	std::vector<double>&        kseedsx,
	std::vector<double>&        kseedsy,
	std::vector<double>&        nlabel,
	unsigned int*			ubuff,
	int*						labels,
	const int&					width,
	const int&					height,
	cv::Scalar color
	)

{
	int clorfazhi(45), piontnumber(4);
	double ab(0), distance(0);
	double colora(0), colorb(0), colordistancel(0), colordistancea(0), colordistanceb(0);
	int zdnum, seednum, labVALUE(0), labVALUE1(0), labVALUE2(0), templab(0);
	int count(0), colorcount1(0), colorcount2(0), colorcount3(0), indexi, indexj;
	zdnum = width*height;
	seednum = kseedsx.size();
	int* nlabels = new int[zdnum];
	std::vector<double>        nlabelcount, nlabelcount1, newnlabelcount1, newnlabelcount2, nlabelscolorsite;
	std::vector<double> clustersize1, numnewnlabelclustersize;

	nlabelcount1.clear();
	nlabelcount.clear();
	newnlabelcount1.clear();
	newnlabelcount2.clear();
	nlabelscolorsite.clear();
	clustersize1.clear();
	numnewnlabelclustersize.clear();



	nlabelcount1.resize(0);
	nlabelcount.resize(0);
	newnlabelcount1.resize(0);
	newnlabelcount2.resize(0);
	nlabelscolorsite.resize(0);
	clustersize1.resize(0);
	numnewnlabelclustersize.resize(0);



	for (int i = 0; i < zdnum; i++)
	{
		nlabels[i] = labels[i];
		nlabelcount1.push_back(labels[i]);

	}


	bool state(true), state1(true), state2(false), state3(false);

	int numcountlabel1(2), numcountlabel(0);




	for (int c = 0; c < 27; c++)
	{
		nlabelcount.clear();
		nlabelcount.resize(0);
		nlabelcount.push_back(0);
		colorcount3 = 0;//从零开始计数


		for (int i = 0; i < seednum; i++)
		{
			colorcount2 = 0;
			for (int j = 0; j < seednum; j++)
			{
				if (i != j)
				{


					ab = ((kseedsx[i] - kseedsx[j])* (kseedsx[i] - kseedsx[j]) + (kseedsy[i] - kseedsy[j])*(kseedsy[i] - kseedsy[j]));
					distance = sqrt(ab);
					if (distance <40)
					{
						indexi = kseedsy[i] * width + kseedsx[i];
						indexj = kseedsy[j] * width + kseedsx[j];

						colordistancel = (kseedsl[i] - kseedsl[j]);
						colordistancea = (kseedsa[i] - kseedsa[j]);
						colordistanceb = (kseedsb[i] - kseedsb[j]);

						if (((colordistancel >clorfazhi) && (colordistancea > clorfazhi) && (colordistanceb && clorfazhi)) || (colordistancel > 125) || (colordistancea > 125) || (colordistanceb > 125))//35
						{
							colorcount2++;
							if (colorcount2 >= piontnumber)//对于40时候可以取5个
							{
								int dex(0), num1(0), num2(0);
								dex = nlabelcount.size();

								num1 = nlabelcount[dex - 1];
								num2 = nlabelcount1[indexi];
								if ((num1 < num2) && (num1 < i))
								{
									//colorcount2 = 0;
									colorcount3++;
									nlabelcount.push_back(i);


								}

							}
						}
						else
							continue;

					}
					else
						continue;
				}
				else
					continue;

			}

		}



		numcountlabel = nlabelcount.size();

		if (numcountlabel >= 8 && clorfazhi > 0 && piontnumber > 0)
		{
			if (clorfazhi < 100)//150
			{
				clorfazhi += 5;
				if (clorfazhi>85 && piontnumber <= 10)
				{
					clorfazhi -= 5;
					piontnumber += 1;
				}
				else
					continue;
			}
			else
				break;


		}
		else if (numcountlabel <= 1 && clorfazhi > 0 && piontnumber > 0)
		{
			if (clorfazhi > 45)
			{
				clorfazhi -= 5;
				if (clorfazhi < 5 && piontnumber >= 4)
				{
					clorfazhi += 5;
					piontnumber -= 1;
				}
				else
					continue;
			}
			else
				break;

		}


		else
			break;
	}



	//std::cout << "当前图像存在" << numcountlabel -1<< "个目标物体" << std::endl;
	for (int n = 1; n < numcountlabel; n++)
	{
		int  testresultlabelvalue = nlabelcount[n];
		int ox = kseedsx[testresultlabelvalue];//original x
		int oy = kseedsy[testresultlabelvalue];//original y
		DrawScalesRects(cv::Point(ox, oy));
	}

	delete nlabels;
}

bool IPSG::SLIC::DrawScalesRects(cv::Point PointXY)
{
	//	int width = 128;
	//	int height = 128;
	m_SOutSamples.m_vOutSamples.clear();
	m_SOutSamples.m_vOutRects.clear();
	m_SOutSamples.m_CenterPoints.clear();

	m_SOutSamples.SaliencyHeight = 96;
	m_SOutSamples.SaliencyWidth = 96;

	if (PointXY.x - m_SOutSamples.SaliencyWidth / 2 < 0)
	{
		m_nRectX = 0;
	}
	else
	{
		m_nRectX = PointXY.x - m_SOutSamples.SaliencyWidth / 2;
	}

	if (PointXY.y - m_SOutSamples.SaliencyHeight / 2 < 0)
	{
		m_nRectY = 0;
	}
	else
	{
		m_nRectY = PointXY.y - m_SOutSamples.SaliencyHeight / 2;
	}

	cv::Rect Rect1(m_nRectX, m_nRectY, m_SOutSamples.SaliencyWidth, m_SOutSamples.SaliencyHeight); //128*128
	//cv::Rect Rect2(PointXY.x - width / 2, PointXY.y - height, width, 2 * height); //128*256
	//cv::Rect Rect3(PointXY.x - width, PointXY.y - height / 2, 2 * width, height); //256*128
	//cv::Rect Rect4(PointXY.x - width, PointXY.y - height, 2 * width, 2 * height); //256*256
	cv::Rect RectImg(0, 0, m_SOutSamples.m_ImgSrc.cols, m_SOutSamples.m_ImgSrc.rows);

	m_SOutSamples.m_vOutSamples.push_back(m_SOutSamples.m_ImgSrc(Rect1&RectImg));
	//m_SOutSamples.m_vOutSamples.push_back(m_SOutSamples.m_ImgSrc(Rect2&RectImg));
	//m_SOutSamples.m_vOutSamples.push_back(m_SOutSamples.m_ImgSrc(Rect3&RectImg));
	//m_SOutSamples.m_vOutSamples.push_back(m_SOutSamples.m_ImgSrc(Rect4&RectImg));

	cv::Rect r1 = Rect1&RectImg;
	//cv::Rect r2 = Rect2&RectImg;
	//cv::Rect r3 = Rect3&RectImg;
	//cv::Rect r4 = Rect4&RectImg;

	m_SOutSamples.m_vOutRects.push_back(r1);
	//m_SOutSamples.m_vOutRects.push_back(r2);
	//m_SOutSamples.m_vOutRects.push_back(r3);
	//m_SOutSamples.m_vOutRects.push_back(r4);

	m_SOutSamples.m_CenterPoints.push_back(cv::Point(r1.x + (r1.width / 2), r1.y + (r1.height / 2)));
	//m_SOutSamples.m_CenterPoints.push_back(cv::Point(r2.x + (r2.width / 2), r2.y + (r2.height / 2)));
	//m_SOutSamples.m_CenterPoints.push_back(cv::Point(r3.x + (r3.width / 2), r3.y + (r3.height / 2)));
	//m_SOutSamples.m_CenterPoints.push_back(cv::Point(r4.x + (r4.width / 2), r4.y + (r4.height / 2)));

	cv::Mat ImgClone = m_SOutSamples.m_ImgSrc.clone();
	cv::rectangle(ImgClone, Rect1&RectImg, cv::Scalar(0, 0, 255), 2);
	//cv::rectangle(ImgClone, Rect2&RectImg, cv::Scalar(0, 0, 255), 2);
	//cv::rectangle(ImgClone, Rect3&RectImg, cv::Scalar(0, 0, 255), 2);
	//cv::rectangle(ImgClone, Rect4&RectImg, cv::Scalar(0, 0, 255), 2);
	cv::imshow("ImgRects", ImgClone);
	SaveImage(ImgClone, "E:\\东方水利\\超像素分割后的图\\");
	cv::waitKey(10);
	ImgClone.release();
	return true;
}

void IPSG::SLIC::SaveImage(cv::Mat inputImage, std::string path)
{
	SYSTEMTIME nowtime;
	GetLocalTime(&nowtime);
	cv::Mat saveImage = inputImage.clone();
	std::string strCurrentTime = std::to_string(nowtime.wYear).append("_") +
		std::to_string(nowtime.wMonth).append("_") +
		std::to_string(nowtime.wDay).append("_") +
		std::to_string(nowtime.wHour).append("_") +
		std::to_string(nowtime.wMinute).append("_") +
		std::to_string(nowtime.wSecond);
	std::string saveName = path + strCurrentTime + ".jpg";

	cv::imwrite(saveName, saveImage);
	saveImage.release();
}

bool IPSG::SLIC::SuperpixelTest(cv::Mat Img, int numSuperpixel)
{
	cv::Mat result;
	m_SOutSamples.m_ImgSrc = Img.clone();
	if (!m_SOutSamples.m_ImgSrc.data)
	{
		std::cout << "load image error!!" << std::endl;
		return false;
	}
	std::clock_t clock_begin, clock_end;
	clock_begin = std::clock();

	GenerateSuperpixels(m_SOutSamples.m_ImgSrc, numSuperpixel);

	clock_end = clock();
	//printf("time elapsed: %f (ms), for img size: %dx%d\n", (float)(clock_end - clock_begin) / CLOCKS_PER_SEC * 1000, m_SOutSamples.m_ImgSrc.rows, m_SOutSamples.m_ImgSrc.cols);

	if (m_SOutSamples.m_ImgSrc.channels() == 3)
		result = GetImgWithContours(numSuperpixel, cv::Scalar(0, 0, 255));
	else
		result = GetImgWithContours(numSuperpixel, cv::Scalar(128));
}

bool IPSG::SLIC::SuperpixelTest2(cv::Mat Img, int numSuperpixel)
{
	cv::Mat result;
	m_SOutSamples.m_ImgSrc = Img.clone();
	if (!m_SOutSamples.m_ImgSrc.data)
	{
		std::cout << "load image error!!" << std::endl;
		return false;
	}
	std::clock_t clock_begin, clock_end;
	clock_begin = std::clock();

	GenerateSuperpixels2(m_SOutSamples.m_ImgSrc, numSuperpixel);

	clock_end = clock();
	//printf("time elapsed: %f (ms), for img size: %dx%d\n", (float)(clock_end - clock_begin) / CLOCKS_PER_SEC * 1000, m_SOutSamples.m_ImgSrc.rows, m_SOutSamples.m_ImgSrc.cols);

	if (m_SOutSamples.m_ImgSrc.channels() == 3)
		result = GetImgWithContours(numSuperpixel, cv::Scalar(0, 0, 255));
	else
		result = GetImgWithContours(numSuperpixel, cv::Scalar(128));
}
