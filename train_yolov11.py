
from ultralytics import YOLO
from roboflow import Roboflow
import shutil
import os
import argparse


def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--name", type=str, default="yolov11")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=96)
    return parser.parse_args()

if __name__ == "__main__":

    yolov11_model = YOLO("yolo11n.pt")
    folder_path = "/kaggle/working/Acne04-Detection-2"
    
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("jimmys-workspace-1ktw6").project("acne04-detection-i2hqg")
    version = project.version(2)
    dataset = version.download("yolov11")
     
    results = yolov11_model.train(
        data=f"{dataset.location}/data.yaml", 
        epochs=args.epochs,
        imgsz=640, 
        batch=args.batch_size,       
        patience=args.patience,
        device=device,
        single_cls=True, 
        name=args.name
    )
