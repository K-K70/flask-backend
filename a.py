from ultralytics import YOLO

# YOLOモデルの読み込み
model = YOLO('best.pt')

# モデル情報を表示
print(model.info())
