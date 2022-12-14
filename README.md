# 環境構築
- https://literacyboxes.com/git-install-windows/ を参考にgitをインストール
- ```pip install opencv-python```
- ```pip install cmake```
- ```pip install dlib```
- ```pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113```
  - https://pytorch.org/get-started/locally/#start-locally で適切なコマンドを選択
- ```pip3 install matplotlib``` 
- ```pip install openpyxl```

# ディレクトリ構成
```sh
root/
  ┣  percentage/  # 実験終了後笑顔の割合を計測するときに使用するフォルダ
  ┃     ┣   haarcascade_frontalface_default.xml  # percentage.pyで必要なhaarcascade分類器
  ┃     ┣   haarcascade_smile.xml  # percentage.pyで必要なhaarcascade分類器
  ┃     ┗   percentage.py  # 実験終了後笑顔の割合を測定する
  ┣  experiment/  # 実験中使用するフォルダ
  ┃     ┣   model/  # 顔の特徴点を学習したファイル
  ┃     ┣   img_utils_pytorch.py  # smile.pyで必要なライブラリ
  ┃     ┣   interp_torch.py  # smile.pyで必要なライブラリ
  ┃     ┣   smile.py  # 笑顔化した顔を流す
  ┃     ┣   real1.py  # リアルの顔を流す(pc1で実行する用)  
  ┃     ┣   real2.py  # リアルの顔を流す(pc2で実行する用)
  ┃     ┣   real3.py  # リアルの顔を流す(pc3で実行する用)
  ┃     ┗   real4.py  # リアルの顔を流す(pc4で実行する用)
  ┗  README.md
```