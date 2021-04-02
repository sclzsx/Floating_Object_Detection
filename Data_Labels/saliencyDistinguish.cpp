#include "saliencyDistinguish.h"

bool IPSG::Csaliency::SaliencyDistingush(cv::Mat inputImage)
{
	cv::Mat Image = inputImage.clone();
	slic.SuperpixelTest(Image);
	for (int SaliencyNum = 0; SaliencyNum < slic.m_SOutSamples.m_vOutSamples.size(); SaliencyNum++)
	{
		if (classifier.RecongitionClassifier(slic.m_SOutSamples.m_vOutSamples.at(SaliencyNum)))
		{
			cv::rectangle(Image, slic.m_SOutSamples.m_vOutRects.at(SaliencyNum), cv::Scalar(255, 0, 0), 2);
		}
	}
	Image.release();
	return true;
}