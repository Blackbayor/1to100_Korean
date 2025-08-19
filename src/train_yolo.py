
from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model.")
    parser.add_argument('--data_config', required=True, help='Path to the data.yaml file.')
    parser.add_argument('--pretrained_weights', required=True, help='Path to the pretrained .pt file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training.')

    args = parser.parse_args()

    # Load a model
    # model = YOLO('yolov8n.pt')  # Load a pre-trained model (recommended for training)
    model = YOLO(args.pretrained_weights) # Load a custom-trained model

    # Train the model
    results = model.train(data=args.data_config, epochs=args.epochs, imgsz=args.imgsz)

    print("Training finished.")
    print(f"Results saved to: {results.save_dir}")
