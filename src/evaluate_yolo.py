
from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 model.")
    parser.add_argument('--model_weights', required=True, help='Path to the trained .pt file.')
    parser.add_argument('--data_config', required=True, help='Path to the data.yaml file.')

    args = parser.parse_args()

    # Load the model
    model = YOLO(args.model_weights)

    # Evaluate the model
    metrics = model.val(data=args.data_config)

    print("Evaluation metrics:")
    print(metrics)
