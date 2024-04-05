from ultralytics import YOLO

import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="datasets/data.yaml", batch=-1, epochs=100, save=True, save_period=25, device=0 if torch.cuda.is_available() else "cpu")
    metrics = model.val()


if __name__ == '__main__':
    main()
