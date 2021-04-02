//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include <iostream>
//using namespace std;
//using namespace cv;
//void change(Mat input, Mat &new_image, double alpha, int beta)
//{
//	for (int y = 0; y < input.rows; y++) {
//		for (int x = 0; x < input.cols; x++) {
//			for (int c = 0; c < 3; c++) {
//				new_image.at<Vec3b>(y, x)[c] =
//					saturate_cast<uchar>(alpha*(input.at<Vec3b>(y, x)[c]) + beta);
//			}
//		}
//	}
//}
//int main(int argc, char** argv)
//{
//	//cout << "* Enter the alpha value [1.0-3.0]: "; cin >> alpha;
//	//cout << "* Enter the beta value [0-100]: ";    cin >> beta;
//
//	for (int i= 1;i<=100;i++)
//	{
//		string na = "E:\\lightchange\\test\\" + to_string(i)+".jpg";
//
//		Mat in = imread(na);
//		if (in.empty())
//		{
//			continue;
//		}
//		cout << na << endl;
//
//		vector<Mat> new_images;
//		for (int beta = 0; beta <= 40; beta = beta + 1)
//		{
//			for (double alpha = 1.0; alpha <= 1.2; alpha = alpha + 0.1)
//			{
//				Mat tmp = Mat::zeros(in.size(), in.type());
//				change(in, tmp, alpha, beta);
//				new_images.push_back(tmp);
//			}
//		}
//		//cout << "new_images.size(): " << new_images.size() << endl;
//
//		for (size_t k = 0; k < new_images.size();k++)
//		{
//			//imshow(to_string(k), new_images[k]);
//			//waitKey();
//			string name = "E:\\lightchange\\test_aug\\" +to_string(i)+"_"+to_string(k)+".jpg";
//			imwrite(name, new_images[k]);
//		}
//	}
//
//
//	return 0;
//}