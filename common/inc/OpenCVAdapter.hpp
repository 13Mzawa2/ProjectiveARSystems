
/************************************************************************************************************/
/*										OpenCV Adapter Ver. 1.4												*/
/*																											*/
/*									vizライブラリに対応(2015/5/12)											*/
/*								ピクセル値取得マクロの不備を修正(2015/5/7)									*/
/*								OpenCV用リンカスクリプト 2.2系対応(2014/4/18)								*/
/*			使用方法：includeパス，libパスを指定したのち，OpenCVを使いたい場所でこいつをインクルードする	*/
/*																											*/
/************************************************************************************************************/
#pragma once

#include	<iostream>
#include	<opencv2/opencv.hpp>
//#include	<opencv2/viz.hpp>

using namespace std;
//using namespace cv;

// バージョン取得
#define CV_VERSION_STR CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

// ビルドモード
#ifdef _DEBUG
#define CV_EXT_STR "d.lib"
#else
#define CV_EXT_STR ".lib"
#endif

// ライブラリのリンク（不要な物はコメントアウト）
#pragma comment(lib, "opencv_calib3d"		CV_VERSION_STR CV_EXT_STR)		//	カメラキャリブレーション，ステレオカメラなど
#pragma comment(lib, "opencv_contrib"		CV_VERSION_STR CV_EXT_STR)		//	肌検出，顔認識，MeanShiftなど
#pragma comment(lib, "opencv_core"			CV_VERSION_STR CV_EXT_STR)		//	メモリ確保・解放，行列演算，描画など
#pragma comment(lib, "opencv_features2d"	CV_VERSION_STR CV_EXT_STR)		//	特徴量抽出
//#pragma comment(lib, "opencv_flann"			CV_VERSION_STR CV_EXT_STR)		//	高速最近傍処理（FLANN）など
//#pragma comment(lib, "opencv_gpu"			CV_VERSION_STR CV_EXT_STR)		//	GPUを利用した行列演算・画像処理
#pragma comment(lib, "opencv_highgui"		CV_VERSION_STR CV_EXT_STR)		//	GUI，ファイル処理，カメラなど
#pragma comment(lib, "opencv_imgproc"		CV_VERSION_STR CV_EXT_STR)		//	画像処理（ヒストグラム，フィルタリング，幾何学変換，物体追跡，特徴検出など）
#pragma comment(lib, "opencv_legacy"		CV_VERSION_STR CV_EXT_STR)		//	旧関数との互換用
//#pragma comment(lib, "opencv_ml"			CV_VERSION_STR CV_EXT_STR)		//	SVM等の機械学習
//#pragma comment(lib, "opencv_nonfree"		CV_VERSION_STR CV_EXT_STR)		//	SIFT，SURF等の特許に関係のあるもの
//#pragma comment(lib, "opencv_objdetect"		CV_VERSION_STR CV_EXT_STR)		//	Haar-Like，LBP，HOG分類器等のオブジェクト検出
//#pragma comment(lib, "opencv_ocl"			CV_VERSION_STR CV_EXT_STR)		//	OpenCLライブラリを使用した並列演算
//#pragma comment(lib, "opencv_photo"			CV_VERSION_STR CV_EXT_STR)		//	Computational Photography分野のサポート（画像修復など？）
//#pragma comment(lib, "opencv_stitching"		CV_VERSION_STR CV_EXT_STR)		//	Stitching（パノラマ撮影などの画像結合）
//#pragma comment(lib, "opencv_superres"		CV_VERSION_STR CV_EXT_STR)		//	Super-Resolution（超解像処理：ぼけやノイズの除去に使える）
//#pragma comment(lib, "opencv_ts"			CV_VERSION_STR CV_EXT_STR)		//	ユーザ定義可能なライブラリ（テスト作成用モジュール）
#pragma comment(lib, "opencv_video"			CV_VERSION_STR CV_EXT_STR)		//	ビデオ解析，モーション解析
//#pragma comment(lib, "opencv_videostab"		CV_VERSION_STR CV_EXT_STR)		//	ビデオ安定化
//#pragma comment(lib, "opencv_viz"  CV_VERSION_STR CV_EXT_STR)				//	3次元データ，PointCloud

//IplImageのピクセル値拾得用マクロ 行(Y)ごとの列(X)番目の配列になっている
#define iplB(IMG,X,Y)		((uchar*)((IMG)->imageData+(IMG)->widthStep*(Y)))[(X)*3]
#define iplG(IMG,X,Y)		((uchar*)((IMG)->imageData+(IMG)->widthStep*(Y)))[(X)*3+1]
#define iplR(IMG,X,Y)		((uchar*)((IMG)->imageData+(IMG)->widthStep*(Y)))[(X)*3+2]
#define iplGRAY(IMG,X,Y)	((uchar*)((IMG)->imageData+(IMG)->widthStep*(Y)))[(X)]

//cv::Matのピクセル値拾得用マクロ
#define matB(IMG,X,Y)		((IMG).data[((IMG).step*(Y) + (IMG).channels()*(X)) + 0])
#define matG(IMG,X,Y)		((IMG).data[((IMG).step*(Y) + (IMG).channels()*(X)) + 1])
#define matR(IMG,X,Y)		((IMG).data[((IMG).step*(Y) + (IMG).channels()*(X)) + 2])
#define matGRAY(IMG,X,Y)	matB(IMG,X,Y)
#define matBf(IMG,X,Y)		(((Point3f*)((IMG).data + (IMG).step.p[0] * Y))[X].x)
#define matGf(IMG,X,Y)		(((Point3f*)((IMG).data + (IMG).step.p[0] * Y))[X].y)
#define matRf(IMG,X,Y)		(((Point3f*)((IMG).data + (IMG).step.p[0] * Y))[X].z)

