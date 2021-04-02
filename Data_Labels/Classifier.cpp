#include "Classifier.h"

IPSG::CClassifierMLP::CClassifierMLP()
{

}

IPSG::CClassifierMLP::~CClassifierMLP()
{

}

void IPSG::CClassifierMLP::dev_update_off()
{
	return;
}

void IPSG::CClassifierMLP::disp_continue_message(HTuple hv_WindowHandle, HTuple hv_Color, HTuple hv_Box)
{

	HTuple  hv_GenParamName, hv_GenParamValue, hv_ContinueMessage;

	hv_GenParamName = HTuple();
	hv_GenParamValue = HTuple();
	if (0 != ((hv_Box.TupleLength()) > 0))
	{
		if (0 != (HTuple(hv_Box[0]) == HTuple("false")))
		{
			//Display no box
			hv_GenParamName = hv_GenParamName.TupleConcat("box");
			hv_GenParamValue = hv_GenParamValue.TupleConcat("false");
		}
		else if (0 != (HTuple(hv_Box[0]) != HTuple("true")))
		{
			//Set a color other than the default.
			hv_GenParamName = hv_GenParamName.TupleConcat("box_color");
			hv_GenParamValue = hv_GenParamValue.TupleConcat(HTuple(hv_Box[0]));
		}
	}
	if (0 != ((hv_Box.TupleLength()) > 1))
	{
		if (0 != (HTuple(hv_Box[1]) == HTuple("false")))
		{
			//Display no shadow.
			hv_GenParamName = hv_GenParamName.TupleConcat("shadow");
			hv_GenParamValue = hv_GenParamValue.TupleConcat("false");
		}
		else if (0 != (HTuple(hv_Box[1]) != HTuple("true")))
		{
			//Set a shadow color other than the default.
			hv_GenParamName = hv_GenParamName.TupleConcat("shadow_color");
			hv_GenParamValue = hv_GenParamValue.TupleConcat(HTuple(hv_Box[1]));
		}
	}

	if (0 != (hv_Color == HTuple("")))
	{
		//disp_text does not accept an empty string for Color.
		hv_Color = HTuple();
	}

	//Display the message.
	hv_ContinueMessage = "Press Run (F5) to continue";
	//DispText(hv_WindowHandle, hv_ContinueMessage, "window", "bottom", "right", hv_Color, hv_GenParamName, hv_GenParamValue);
	return;
}

void IPSG::CClassifierMLP::disp_end_of_program_message(HTuple hv_WindowHandle, HTuple hv_Color, HTuple hv_Box)
{
	HTuple  hv_GenParamName, hv_GenParamValue, hv_EndMessage;

	hv_GenParamName = HTuple();
	hv_GenParamValue = HTuple();

	if (0 != ((hv_Box.TupleLength()) > 0))
	{
		if (0 != (HTuple(hv_Box[0]) == HTuple("false")))
		{
			//Display no box
			hv_GenParamName = hv_GenParamName.TupleConcat("box");
			hv_GenParamValue = hv_GenParamValue.TupleConcat("false");
		}
		else if (0 != (HTuple(hv_Box[0]) != HTuple("true")))
		{
			//Set a color other than the default.
			hv_GenParamName = hv_GenParamName.TupleConcat("box_color");
			hv_GenParamValue = hv_GenParamValue.TupleConcat(HTuple(hv_Box[0]));
		}
	}
	if (0 != ((hv_Box.TupleLength()) > 1))
	{
		if (0 != (HTuple(hv_Box[1]) == HTuple("false")))
		{
			//Display no shadow.
			hv_GenParamName = hv_GenParamName.TupleConcat("shadow");
			hv_GenParamValue = hv_GenParamValue.TupleConcat("false");
		}
		else if (0 != (HTuple(hv_Box[1]) != HTuple("true")))
		{
			//Set a shadow color other than the default.
			hv_GenParamName = hv_GenParamName.TupleConcat("shadow_color");
			hv_GenParamValue = hv_GenParamValue.TupleConcat(HTuple(hv_Box[1]));
		}
	}

	if (0 != (hv_Color == HTuple("")))
	{
		//disp_text does not accept an empty string for Color.
		hv_Color = HTuple();
	}

	//Display the message.
	hv_EndMessage = "      End of program      ";
	//DispText(hv_WindowHandle, hv_EndMessage, "window", "bottom", "right", hv_Color, 
	//    hv_GenParamName, hv_GenParamValue);
	return;
}

void IPSG::CClassifierMLP::disp_message(HTuple hv_WindowHandle, HTuple hv_String, HTuple hv_CoordSystem, HTuple hv_Row, HTuple hv_Column, HTuple hv_Color, HTuple hv_Box)
{
	HTuple  hv_GenParamName, hv_GenParamValue;

	if (0 != (HTuple(hv_Row == HTuple()).TupleOr(hv_Column == HTuple())))
	{
		return;
	}
	if (0 != (hv_Row == -1))
	{
		hv_Row = 12;
	}
	if (0 != (hv_Column == -1))
	{
		hv_Column = 12;
	}

	//Convert the parameter Box to generic parameters.
	hv_GenParamName = HTuple();
	hv_GenParamValue = HTuple();
	if (0 != ((hv_Box.TupleLength()) > 0))
	{
		if (0 != (HTuple(hv_Box[0]) == HTuple("false")))
		{
			//Display no box
			hv_GenParamName = hv_GenParamName.TupleConcat("box");
			hv_GenParamValue = hv_GenParamValue.TupleConcat("false");
		}
		else if (0 != (HTuple(hv_Box[0]) != HTuple("true")))
		{
			//Set a color other than the default.
			hv_GenParamName = hv_GenParamName.TupleConcat("box_color");
			hv_GenParamValue = hv_GenParamValue.TupleConcat(HTuple(hv_Box[0]));
		}
	}
	if (0 != ((hv_Box.TupleLength()) > 1))
	{
		if (0 != (HTuple(hv_Box[1]) == HTuple("false")))
		{
			//Display no shadow.
			hv_GenParamName = hv_GenParamName.TupleConcat("shadow");
			hv_GenParamValue = hv_GenParamValue.TupleConcat("false");
		}
		else if (0 != (HTuple(hv_Box[1]) != HTuple("true")))
		{
			//Set a shadow color other than the default.
			hv_GenParamName = hv_GenParamName.TupleConcat("shadow_color");
			hv_GenParamValue = hv_GenParamValue.TupleConcat(HTuple(hv_Box[1]));
		}
	}
	//Restore default CoordSystem behavior.
	if (0 != (hv_CoordSystem != HTuple("window")))
	{
		hv_CoordSystem = "image";
	}

	if (0 != (hv_Color == HTuple("")))
	{
		//disp_text does not accept an empty string for Color.
		hv_Color = HTuple();
	}

	//DispText(hv_WindowHandle, hv_String, hv_CoordSystem, hv_Row, hv_Column, hv_Color, 
	//    hv_GenParamName, hv_GenParamValue);
	return;
}

void IPSG::CClassifierMLP::set_display_font(HTuple hv_WindowHandle, HTuple hv_Size, HTuple hv_Font, HTuple hv_Bold, HTuple hv_Slant)
{
	HTuple  hv_OS, hv_Fonts, hv_Style, hv_Exception;
	HTuple  hv_AvailableFonts, hv_Fdx, hv_Indices;

	GetSystem("operating_system", &hv_OS);

	if (0 != (HTuple(hv_Size == HTuple()).TupleOr(hv_Size == -1)))
	{
		hv_Size = 16;
	}
	if (0 != ((hv_OS.TupleSubstr(0, 2)) == HTuple("Win")))
	{
		//Restore previous behaviour
		hv_Size = (1.13677*hv_Size).TupleInt();
	}
	if (0 != (hv_Font == HTuple("Courier")))
	{
		hv_Fonts.Clear();
		hv_Fonts[0] = "Courier";
		hv_Fonts[1] = "Courier 10 Pitch";
		hv_Fonts[2] = "Courier New";
		hv_Fonts[3] = "CourierNew";
	}
	else if (0 != (hv_Font == HTuple("mono")))
	{
		hv_Fonts.Clear();
		hv_Fonts[0] = "Consolas";
		hv_Fonts[1] = "Menlo";
		hv_Fonts[2] = "Courier";
		hv_Fonts[3] = "Courier 10 Pitch";
		hv_Fonts[4] = "FreeMono";
	}
	else if (0 != (hv_Font == HTuple("sans")))
	{
		hv_Fonts.Clear();
		hv_Fonts[0] = "Luxi Sans";
		hv_Fonts[1] = "DejaVu Sans";
		hv_Fonts[2] = "FreeSans";
		hv_Fonts[3] = "Arial";
	}
	else if (0 != (hv_Font == HTuple("serif")))
	{
		hv_Fonts.Clear();
		hv_Fonts[0] = "Times New Roman";
		hv_Fonts[1] = "Luxi Serif";
		hv_Fonts[2] = "DejaVu Serif";
		hv_Fonts[3] = "FreeSerif";
		hv_Fonts[4] = "Utopia";
	}
	else
	{
		hv_Fonts = hv_Font;
	}

	hv_Style = "";
	if (0 != (hv_Bold == HTuple("true")))
	{
		hv_Style += HTuple("Bold");
	}
	else if (0 != (hv_Bold != HTuple("false")))
	{
		hv_Exception = "Wrong value of control parameter Bold";
		throw HalconCpp::HException(hv_Exception);
	}
	if (0 != (hv_Slant == HTuple("true")))
	{
		hv_Style += HTuple("Italic");
	}
	else if (0 != (hv_Slant != HTuple("false")))
	{
		hv_Exception = "Wrong value of control parameter Slant";
		throw HalconCpp::HException(hv_Exception);
	}
	if (0 != (hv_Style == HTuple("")))
	{
		hv_Style = "Normal";
	}

	QueryFont(hv_WindowHandle, &hv_AvailableFonts);
	hv_Font = "";

	{
		HTuple end_val48 = (hv_Fonts.TupleLength()) - 1;
		HTuple step_val48 = 1;
		for (hv_Fdx = 0; hv_Fdx.Continue(end_val48, step_val48); hv_Fdx += step_val48)
		{
			hv_Indices = hv_AvailableFonts.TupleFind(HTuple(hv_Fonts[hv_Fdx]));
			if (0 != ((hv_Indices.TupleLength()) > 0))
			{
				if (0 != (HTuple(hv_Indices[0]) >= 0))
				{
					hv_Font = HTuple(hv_Fonts[hv_Fdx]);
					break;
				}
			}
		}
	}

	if (0 != (hv_Font == HTuple("")))
	{
		throw HalconCpp::HException("Wrong value of control parameter Font");
	}

	hv_Font = (((hv_Font + "-") + hv_Style) + "-") + hv_Size;
	SetFont(hv_WindowHandle, hv_Font);

	return;
}

void IPSG::CClassifierMLP::gen_features(HObject ho_Image, HTuple *hv_FeatureVector)
{
	HObject  ho_Zoomed1;

	(*hv_FeatureVector) = HTuple();
	//Compute features.
	gen_sobel_features(ho_Image, (*hv_FeatureVector), &(*hv_FeatureVector));
	//Downscale the image (image pyramid) and compute features.
	ZoomImageFactor(ho_Image, &ho_Zoomed1, 0.5, 0.5, "constant");
	gen_sobel_features(ho_Zoomed1, (*hv_FeatureVector), &(*hv_FeatureVector));
	//Uncomment lines to use further pyramid levels:
	//zoom_image_factor (Zoomed1, Zoomed2, 0.5, 0.5, 'constant')
	//gen_sobel_features (Zoomed2, FeatureVector, FeatureVector)
	//zoom_image_factor (Zoomed2, Zoomed3, 0.5, 0.5, 'constant')
	//gen_sobel_features (Zoomed3, FeatureVector, FeatureVector)
	//zoom_image_factor (Zoomed3, Zoomed4, 0.5, 0.5, 'constant')
	//gen_sobel_features (Zoomed4, FeatureVector, FeatureVector)
	(*hv_FeatureVector) = (*hv_FeatureVector).TupleReal();
	return;
}

void IPSG::CClassifierMLP::gen_sobel_features(HObject ho_Image, HTuple hv_Features, HTuple *hv_FeaturesExtended)
{
	// Local iconic variables
	HObject  ho_EdgeAmplitude;

	// Local control variables
	HTuple  hv_Energy, hv_Correlation, hv_Homogeneity;
	HTuple  hv_Contrast, hv_AbsoluteHistoEdgeAmplitude, hv_Entropy;
	HTuple  hv_Anisotropy, hv_AbsoluteHistoImage;

	//Coocurrence matrix for 90 deg:
	CoocFeatureImage(ho_Image, ho_Image, 6, 90, &hv_Energy, &hv_Correlation, &hv_Homogeneity, &hv_Contrast);
	//Absolute histogram of edge amplitudes:
	SobelAmp(ho_Image, &ho_EdgeAmplitude, "sum_abs", 3);
	GrayHistoAbs(ho_EdgeAmplitude, ho_EdgeAmplitude, 8, &hv_AbsoluteHistoEdgeAmplitude);

	//You could of course compute more features:
	//Entropy and anisotropy:
	EntropyGray(ho_Image, ho_Image, &hv_Entropy, &hv_Anisotropy);
	//Absolute histogram of gray values:
	GrayHistoAbs(ho_Image, ho_Image, 8, &hv_AbsoluteHistoImage);
	//Add features to feature vector:
	(*hv_FeaturesExtended).Clear();
	(*hv_FeaturesExtended).Append(hv_Features);
	(*hv_FeaturesExtended).Append(hv_Energy);
	(*hv_FeaturesExtended).Append(hv_Correlation);
	(*hv_FeaturesExtended).Append(hv_Homogeneity);
	(*hv_FeaturesExtended).Append(hv_Contrast);
	(*hv_FeaturesExtended) = (*hv_FeaturesExtended).TupleConcat(hv_AbsoluteHistoEdgeAmplitude);
	//Activate the following lines to add the additional features you activated:
	(*hv_FeaturesExtended) = ((*hv_FeaturesExtended).TupleConcat(hv_Entropy)).TupleConcat(hv_Anisotropy);
	(*hv_FeaturesExtended) = (*hv_FeaturesExtended).TupleConcat(hv_AbsoluteHistoImage);
	return;
}

bool IPSG::CClassifierMLP::MatToHImage(cv::Mat& InputImage, HObject& HSrcImage)
{
	if (InputImage.empty())
	{
		return false;
	}

	if (InputImage.channels() == 1)
	{
		int height = InputImage.rows;
		int width = InputImage.cols;
		uchar *dataGray = new uchar[width*height];
		for (int i = 0; i < height; i++)
		{
			memcpy(dataGray + width*i, InputImage.data + InputImage.step*i, width);
		}
		GenImage1(&HSrcImage, "byte", InputImage.cols, InputImage.rows, (Hlong)(dataGray));
		delete[] dataGray;
	}
	if (InputImage.channels() == 3)
	{
		int height = InputImage.rows;
		int width = InputImage.cols;
		cv::Mat  ImageRed, ImageGreen, ImageBlue;
		ImageRed = cv::Mat(height, width, CV_8UC1);
		ImageGreen = cv::Mat(height, width, CV_8UC1);
		ImageBlue = cv::Mat(height, width, CV_8UC1);
		std::vector<cv::Mat> ImageChannels;
		split(InputImage, ImageChannels);

		ImageBlue = ImageChannels.at(0);
		ImageGreen = ImageChannels.at(1);
		ImageRed = ImageChannels.at(2);

		uchar*  dataRed = new uchar[InputImage.cols*InputImage.rows];
		uchar*  dataGreen = new uchar[InputImage.cols*InputImage.rows];
		uchar*  dataBlue = new uchar[InputImage.cols*InputImage.rows];
		for (int i = 0; i < height; i++)
		{
			memcpy(dataRed + width*i, ImageRed.data + ImageRed.step*i, width);
			memcpy(dataGreen + width*i, ImageGreen.data + ImageGreen.step*i, width);
			memcpy(dataBlue + width*i, ImageBlue.data + ImageBlue.step*i, width);
		}
		GenImage3(&HSrcImage, "byte", InputImage.cols, InputImage.rows, (Hlong)(dataRed), (Hlong)(dataGreen), (Hlong)(dataBlue));
		delete[]  dataRed;
		delete[]  dataGreen;
		delete[]  dataBlue;
	}
	return true;
}

bool IPSG::CClassifierMLP::RecongitionClassifier(cv::Mat& InputImage, bool &Category)
{
	HObject  ho_Image, ho_Image1;

	HTuple  hv_FileExists, hv_USE_STORED_CLASSIFIER;
	HTuple  hv_Classes, hv_MLPHandle, hv_NumClasses;
	HTuple  hv_FeatureVector;
	HTuple  hv_CorrectClassID;
	HTuple  hv_FoundClassIDs, hv_Confidence;
	HTuple  hv_Width2, hv_Height2;

	MatToHImage(InputImage, ho_Image);
	FileExists("cefenEdition.gmc", &hv_FileExists);

	if (0 != hv_FileExists)
	{
		hv_USE_STORED_CLASSIFIER = 1;
		std::cout << "Existing training model£¡" << std::endl;
	}
	else
	{
		hv_USE_STORED_CLASSIFIER = 0;
		std::cout << "No model!" << std::endl;
	}

	if (0 != (hv_USE_STORED_CLASSIFIER == 1))
	{
		ReadClassMlp("cefenEdition.gmc", &hv_MLPHandle);
		std::cout << "Read model success£¡" << std::endl;

		hv_NumClasses = hv_Classes.TupleLength();
	}
	else
	{
		std::cout << "Failure to read the model!" << std::endl;
		return false;
	}

	//*****************************************ÊÔ¼þÑù±¾²âÊÔ½×¶Î************************************************

	hv_Classes.Clear();
	hv_Classes[0] = "Neg";
	hv_Classes[1] = "Pos";

	GetImageSize(ho_Image, &hv_Width2, &hv_Height2);
	Emphasize(ho_Image, &ho_Image1, hv_Width2, hv_Height2, 1.0);
	gen_features(ho_Image1, &hv_FeatureVector);
	ClassifyClassMlp(hv_MLPHandle, hv_FeatureVector, 2, &hv_FoundClassIDs, &hv_Confidence);
	//0 != (hv_CorrectClassID == HTuple(hv_FoundClassIDs[0]))
	if (hv_FoundClassIDs[0] == 1)
	{
		Category = true;
		std::cout << "Discovery of floating objects!" << std::endl;

	}
	else
	{
		Category = false;
		std::cout << "no floating objects!" << std::endl;
	}
	ho_Image.Clear();
	InputImage.release();

	return Category;
}

bool IPSG::CClassifierMLP::RecongitionClassifier(cv::Mat src_InputImage)
{
	HObject  ho_Image, ho_Image1;

	HTuple  hv_FileExists, hv_USE_STORED_CLASSIFIER;
	HTuple  hv_Classes, hv_MLPHandle, hv_NumClasses;
	HTuple  hv_FeatureVector;
	HTuple  hv_CorrectClassID;
	HTuple  hv_FoundClassIDs, hv_Confidence;
	HTuple  hv_Width2, hv_Height2;

	cv::Mat InputImage = src_InputImage.clone();
	MatToHImage(InputImage, ho_Image);
	FileExists("cefenEdition.gmc", &hv_FileExists);

	if (0 != hv_FileExists)
	{
		hv_USE_STORED_CLASSIFIER = 1;
	//	std::cout << "Existing training model£¡" << std::endl;
	}
	else
	{
		hv_USE_STORED_CLASSIFIER = 0;
		std::cout << "No model!" << std::endl;
	}

	if (0 != (hv_USE_STORED_CLASSIFIER == 1))
	{
		ReadClassMlp("cefenEdition.gmc", &hv_MLPHandle);
	//	std::cout << "Read model success£¡" << std::endl;

		hv_NumClasses = hv_Classes.TupleLength();
	}
	else
	{
		std::cout << "Failure to read the model!" << std::endl;
		return false;
	}

	//*****************************************ÊÔ¼þÑù±¾²âÊÔ½×¶Î************************************************

	hv_Classes.Clear();
	hv_Classes[0] = "Neg";
	hv_Classes[1] = "Pos";

	GetImageSize(ho_Image, &hv_Width2, &hv_Height2);
	Emphasize(ho_Image, &ho_Image1, hv_Width2, hv_Height2, 1.0);
	gen_features(ho_Image1, &hv_FeatureVector);
	ClassifyClassMlp(hv_MLPHandle, hv_FeatureVector, 2, &hv_FoundClassIDs, &hv_Confidence);
	//0 != (hv_CorrectClassID == HTuple(hv_FoundClassIDs[0]))
	if (hv_FoundClassIDs[0] == 1)
	{
		SaliencyFlag = true;
	//	std::cout << "Discovery of floating objects!" << std::endl;

	}
	else
	{
		SaliencyFlag = false;
	//	std::cout << "no floating objects!" << std::endl;
	}

	ho_Image.Clear();
	ho_Image1.Clear();
	InputImage.release();

	return SaliencyFlag;
}