#version 330 core
 
// インプット頂点データ。このシェーダの実行ごとに異なります。
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec2 vertexUV;
layout(location = 2) in vec3 vertexNormal;
 
// アウトプットデータ。各フラグメントで書き込まれます。
out vec4 vertexPosition_cameraspace;
out vec2 UV;
out vec3 Normal;		//	カメラ座標系での法線
 
// すべてのメッシュで一定の値
uniform mat4 MVP;
uniform mat4 MV;		//	法線計算に使う
 
void main()
{
    // クリップ空間での頂点の出力位置。MVP×位置
    gl_Position =  MVP * vec4(vertexPosition_modelspace,1);
	vertexPosition_cameraspace = MV * vec4(vertexPosition_modelspace, 1);

    // 頂点のUV座標です。特別な空間はありません。
    UV = vertexUV;

	//	法線ベクトルをカメラ行列に変換して渡す
	Normal = vec3(normalize(MV * vec4(vertexNormal, 0)));
}