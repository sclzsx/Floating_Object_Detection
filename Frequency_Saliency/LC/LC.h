#include "opencv2/core/core.hpp"  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgproc/imgproc_c.h>
using namespace cv;
class LC
{
public:
	LC()
	{
	}

	~LC()
	{
	}

public:
	void calculateSaliencyMap(Mat *src, Mat * dst);

};

