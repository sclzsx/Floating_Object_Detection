//#include <iostream>
//#include <vector>
//#include <fstream>
//#include <opencv2/opencv.hpp>
//#include <string>
//using namespace std;
//using namespace cv;
//std::fstream tr;
//std::fstream te;
//
//void printtr(int n, string name)
//{
//	name = "  <image file='" + name + "'>";
//	switch (n)
//	{
//	case 1:tr << name << endl;
//		tr << "    <box top='162' left='181' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 2:tr << name << endl;
//		tr << "    <box top='231' left='158' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 3:tr << name << endl;
//		tr << "    <box top='153' left='141' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 4:tr << name << endl;
//		tr << "    <box top='270' left='57' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 5:tr << name << endl;
//		tr << "    <box top='270' left='25' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 6:tr << name << endl;
//		tr << "    <box top='224' left='108' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 7:tr << name << endl;
//		tr << "    <box top='270' left='92' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 8:tr << name << endl;
//		tr << "    <box top='86' left='71' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 9:tr << name << endl;
//		tr << "    <box top='165' left='48' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 10:tr << name << endl;
//		tr << "    <box top='86' left='42' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 11:tr << name << endl;
//		tr << "    <box top='135' left='57' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 12:tr << name << endl;
//		tr << "    <box top='226' left='55' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 13:tr << name << endl;
//		tr << "    <box top='270' left='23' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 14:tr << name << endl;
//		tr << "    <box top='186' left='242' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 15:tr << name << endl;
//		tr << "    <box top='157' left='249' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 16:tr << name << endl;
//		tr << "    <box top='214' left='224' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 17:tr << name << endl;
//		tr << "    <box top='270' left='83' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 18:tr << name << endl;
//		tr << "    <box top='76' left='173' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 19:tr << name << endl;
//		tr << "    <box top='49' left='181' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 20:tr << name << endl;
//		tr << "    <box top='102' left='153' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 21:tr << name << endl;
//		tr << "    <box top='215' left='181' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 22:tr << name << endl;
//		tr << "    <box top='182' left='189' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 23:tr << name << endl;
//		tr << "    <box top='188' left='145' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 24:tr << name << endl;
//		tr << "    <box top='135' left='262' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 25:tr << name << endl;
//		tr << "    <box top='145' left='270' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 26:tr << name << endl;
//		tr << "    <box top='184' left='258' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 27:tr << name << endl;
//		tr << "    <box top='199' left='249' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 28:tr << name << endl;
//		tr << "    <box top='53' left='270' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 29:tr << name << endl;
//		tr << "    <box top='84' left='270' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 30:tr << name << endl;
//		tr << "    <box top='107' left='151' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 31:tr << name << endl;
//		tr << "    <box top='239' left='184' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 32:tr << name << endl;
//		tr << "    <box top='270' left='162' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 33:tr << name << endl;
//		tr << "    <box top='270' left='178' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 34:tr << name << endl;
//		tr << "    <box top='212' left='127' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 35:tr << name << endl;
//		tr << "    <box top='270' left='148' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 36:tr << name << endl;
//		tr << "    <box top='270' left='115' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 37:tr << name << endl;
//		tr << "    <box top='270' left='100' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 38:tr << name << endl;
//		tr << "    <box top='26' left='256' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 39:tr << name << endl;
//		tr << "    <box top='2' left='270' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 40:tr << name << endl;
//		tr << "    <box top='2' left='236' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	case 41:tr << name << endl;
//		tr << "    <box top='63' left='246' width='80' height='80'/>" << endl;
//		tr << "  </image>" << endl; break;
//	default:break;
//	}
//}
//
//void printte(int n, string name)
//{
//	name = "  <image file='" + name + "'>";
//	switch (n)
//	{
//	case 1:te << name << endl;
//		te << "    <box top='155' left='174' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 2:te << name << endl;
//		te << "    <box top='229' left='132' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 3:te << name << endl;
//		te << "    <box top='270' left='54' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 4:te << name << endl;
//		te << "    <box top='215' left='98' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 5:te << name << endl;
//		te << "    <box top='270' left='67' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 6:te << name << endl;
//		te << "    <box top='110' left='65' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 7:te << name << endl;
//		te << "    <box top='109' left='54' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 8:te << name << endl;
//		te << "    <box top='156' left='50' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 9:te << name << endl;
//		te << "    <box top='267' left='17' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 10:te << name << endl;
//		te << "    <box top='224' left='8' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 11:te << name << endl;
//		te << "    <box top='197' left='213' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 12:te << name << endl;
//		te << "    <box top='155' left='198' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 13:te << name << endl;
//		te << "    <box top='270' left='28' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 14:te << name << endl;
//		te << "    <box top='93' left='140' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 15:te << name << endl;
//		te << "    <box top='226' left='154' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 16:te << name << endl;
//		te << "    <box top='165' left='270' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 17:te << name << endl;
//		te << "    <box top='81' left='270' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 18:te << name << endl;
//		te << "    <box top='92' left='270' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 19:te << name << endl;
//		te << "    <box top='270' left='204' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 20:te << name << endl;
//		te << "    <box top='270' left='116' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	case 21:te << name << endl;
//		te << "    <box top='11' left='257' width='80' height='80'/>" << endl;
//		te << "  </image>" << endl; break;
//	default:break;
//	}
//}
//
//
//
//int main(int argc, char** argv)
//{
//	tr.open("tr.xml", std::ios::app);
//	std::ofstream clctr("tr.xml", std::ios::out);
//
//	te.open("te.xml", std::ios::app);
//	std::ofstream clcte("te.xml", std::ios::out);
//
//	tr << "<?xml version='1.0' encoding='ISO-8859-1'?>" << std::endl;
//	tr << "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>" << std::endl;
//	tr << "<dataset>" << std::endl;
//	tr << "<name>imglab dataset</name>" << std::endl;
//	tr << "<comment>Created by imglab tool.</comment>" << std::endl;
//	tr << "<images>" << std::endl;
//
//	te << "<?xml version='1.0' encoding='ISO-8859-1'?>" << std::endl;
//	te << "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>" << std::endl;
//	te << "<dataset>" << std::endl;
//	te << "<name>imglab dataset</name>" << std::endl;
//	te << "<comment>Created by imglab tool.</comment>" << std::endl;
//	te << "<images>" << std::endl;
//
//	int batch = 1;
//	for (int n = 1; n <= 3362; n++)
//	{
//		string name = "E:\\lightchange\\train_aug\\" + to_string(n) + ".jpg";
//		cout << "current num is " << n << "         current batch is " << batch << endl;
//		printtr(batch, name);
//		if (n % 82 == 0)
//		{
//			batch++;
//		}
//	}
//
//	int batch2 = 1;
//	for (int n = 1; n <= 1722; n++)
//	{
//		string name = "E:\\lightchange\\test_aug\\" + to_string(n) + ".jpg";
//		cout << "current num is " << n << "         current batch is " << batch2 << endl;
//		printte(batch2, name);
//		if (n % 82 == 0)
//		{
//			batch2++;
//		}
//	}
//
//	tr << "</images>" << std::endl;
//	tr << "</dataset>" << std::endl;
//	tr.close();
//
//	te << "</images>" << std::endl;
//	te << "</dataset>" << std::endl;
//	te.close();
//
//	system("pause");
//	return 0;
//}
//
