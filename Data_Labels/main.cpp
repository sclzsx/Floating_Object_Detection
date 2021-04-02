#include "saliencyDistinguish.h"

bool load_images(const std::string dirname, std::vector< cv::Mat >& img_lst, bool gray = 0)
{
	std::vector< std::string > files;
	cv::glob(dirname, files);
	for (size_t i = 0; i < files.size(); ++i)
	{
		cv::Mat img = cv::imread(files[i]); // load the image
		if (img.empty())            // invalid image, skip it.
		{
			std::cout << files[i] << " is invalid!" << std::endl;
			continue;
		}
		if (gray)
		{
			cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		}
		img_lst.push_back(img);
	}
	return false;
}

int main(int argc, char** argv)
{
	const std::string path = "E:\\database\\private\\litter\\src_V2\\detect\\";


	IPSG::Csaliency saliency;

	std::vector<cv::Mat> imgs;
	load_images(path, imgs);
	for (size_t i = 0; i < imgs.size(); i++)
	{
		saliency.SaliencyDistingush(imgs[i]);
		cv::imshow("result", imgs[i]);
		cv::waitKey(0);
	}


	return 0;
}