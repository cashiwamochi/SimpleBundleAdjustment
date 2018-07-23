# Bundle Adjustment 2-View

This repository includes a C++ implementation which performs Bundle Adjustment in 2-View. This BA is very simple, so it maybe a tutorial for beginners. This supports *pose-only-BA, structure-only-BA, full-BA*.
These reference pdf are very helpful.

```
[1]http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-A13-BundleAdjustment.pdf   
[2]https://pdfs.semanticscholar.org/6081/417b95bec070fb842a704044def427f8ef69.pdf   
[3]http://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf   
[Japanese]
[4]https://ipsj.ixsq.nii.ac.jp/ej/?action=repository_action_common_download&item_id=62864&item_no=1&attribute_id=1&file_no=1
```

## Implementation
Using Gauss-Newton method, the projection-error is minimized.

```
BundleAdjustment2Viewes.* -> Implementation of BA.
main_ba.cc -> An example shows how to use BA.
```

### Dependencies

```OpenCV > 3.0```   
OpenCV is used for initialization and calculation of matrices.


### Build & Use
```
mkdir build && cmake .. && make
./ba_example [path/to/image1] [path/to/image2]
```

#### Attension
I cannot make sure that this implementation is currect. However, this system minimizes reprojection-error and looks working correctly in my test. If you notice something wrong, would you tell me isseus or problems?

------
# 2-Viewでのバンドル調整

勉強がてら，2-Viewにおけるバンドル調整を実装した．*pose-only-BA, structure-only-BA, full-BA* を一通りサポートしている．実装にあたっては下記資料が大変参考になった．

```
[1]http://frc.ri.cmu.edu/~kaess/vslam_cvpr14/media/VSLAM-Tutorial-CVPR14-A13-BundleAdjustment.pdf   
[2]https://pdfs.semanticscholar.org/6081/417b95bec070fb842a704044def427f8ef69.pdf   
[3]http://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf   
[日本語]
[4]https://ipsj.ixsq.nii.ac.jp/ej/?action=repository_action_common_download&item_id=62864&item_no=1&attribute_id=1&file_no=1
```

## 実装
まず，2枚の画像で特徴点を対応づけ，カメラ姿勢の推定，そして三角測量を行う．こうして得られたカメラ姿勢，三次元点のどちらか，あるいは両方にノイズを加え意図的に再投影誤差を発生させる(なお，ノイズを与えなくても持っている初期値には改善の余地があるため多少誤差を下げられる)．その後，更新したい成分についてヤコビアンを計算し，ガウスニュートン法によって再投影誤差を最小化するという一般的な流れ．カメラ姿勢や三次元点のパラメタライズには特に ***"A tutorial on SE(3) transformation parameterizations and on-manifold optimization"[3]*** を参考にした．

```
BundleAdjustment2Viewes.* -> バンドル調整の実装
ba_example.cc -> 2画像の特徴点対応づけ，カメラ姿勢推定，三角測量を行い，その後バンドル調整を行う．
```

### 依存

```OpenCV > 3.0```   
OpenCVは初期化と行列計算のために必要．

### ビルド
```
mkdir build && cmake .. && make
./ba_example [path/to/image1] [path/to/image2]
```

#### 注意
この実装が正しいのかわからない．ただし，確認した範囲では実装に誤りはなく，また手元の実験でも正しく再投影誤差を小さくできているため問題はないと思われる．もし，誤りを見つけたならば，教えてください．
