# AI_STYLIST_hair

```
pip install -r requirements.txt
```
https://drive.google.com/drive/folders/1qFjU_GfB8X0w2LY8z9XZHeZrrjZuJgkk?usp=sharing
から、データセットをダウンロードしてルートディレクトリに配置


### モデル髪型の前処理
* 特徴点の抽出
* 髪型の切り抜き
```
python process_model/modelShapePredictor.py 
```

### ユーザーの処理
* モデルの髪型を合成できるように処理
```
python process_user/faceShapePredictor.py  
```
