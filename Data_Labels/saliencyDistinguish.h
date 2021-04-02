#ifndef SALIENCYDISTINGUISH_H_
#define SALIENCYDISTINGUISH_H_

#include "slic.h"
#include "Classifier.h"
#include <vector>

namespace IPSG
{
	class Csaliency
	{
	public:
		Csaliency()
		{
		}
		~Csaliency()
		{}

		bool SaliencyDistingush(cv::Mat inputImage);
	private:
		SLIC slic;
		CClassifierMLP classifier;
	};
}
#endif // !SALIENCYDISTINGUISH_H_
