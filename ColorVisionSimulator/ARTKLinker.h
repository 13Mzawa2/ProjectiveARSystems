#pragma once

#include <AR\ar.h>
//#include <AR\gsub_lite.h>

#ifdef _DEBUG
#define AR_EXT "d.lib"
#else
#define AR_EXT ".lib"
#endif

#pragma comment(lib, "libAR" AR_EXT)
//#pragma comment(lib, "libARgsub_lite" AR_EXT)

#pragma comment(lib, "opengl32.lib")

//	ARToolKit Marker Information Structure
typedef struct ARTKMarker
{
	char *patt_name;			//	パターンファイル名
	int patt_id;				//	パターンID
	int mark_id;				//	マーカーID
	int visible;				//	検出フラグ
	double patt_width;			//	パターンサイズ(mm)
	double patt_center[2];		//	パターン中央座標
	double patt_trans[3][4];	//	座標変換行列
};