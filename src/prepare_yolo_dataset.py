import os
import json
import random
import shutil
from tqdm import tqdm
import argparse

def convert_sagemaker_to_yolo(manifest_files, image_dirs, output_dir, class_map, train_split):
    """
    Converts SageMaker Ground Truth annotations to YOLOv8 format.
    """
    # Create output directories
    dirs = {
        "train_images": os.path.join(output_dir, "images", "train"),
        "val_images": os.path.join(output_dir, "images", "val"),
        "train_labels": os.path.join(output_dir, "labels", "train"),
        "val_labels": os.path.join(output_dir, "labels", "val"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    all_annotations = []

    print("Parsing manifest files and verifying images...")
    for manifest_file, image_dir in zip(manifest_files, image_dirs):
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                # --- Find the correct job name ---
                job_name = None
                for key, value in data.items():
                    if isinstance(value, dict) and 'annotations' in value:
                        job_name = key
                        break
                
                if not job_name:
                    print(f"Warning: Could not find a job with 'annotations' in line from {manifest_file}. Skipping.")
                    continue

                source_ref = data['source-ref']
                image_filename = os.path.basename(source_ref)
                image_path = os.path.join(image_dir, image_filename)

                if not os.path.exists(image_path):
                    print(f"Warning: Image not found, skipping: {image_path}")
                    continue

                image_size_data = data[job_name]['image_size'][0]
                img_width = image_size_data['width']
                img_height = image_size_data['height']
                
                annotations = data[job_name].get('annotations', [])
                
                metadata_key = f'{job_name}-metadata'
                if metadata_key not in data:
                    print(f"Warning: Metadata key '{metadata_key}' not found. Skipping line.")
                    continue

                manifest_class_map = data[metadata_key]['class-map']

                yolo_annotations = []
                for ann in annotations:
                    class_id_str = str(ann['class_id'])
                    class_name = manifest_class_map.get(class_id_str)

                    if class_name is None:
                        print(f"Warning: class_id '{class_id_str}' not found in class-map for {image_filename}. Skipping annotation.")
                        continue

                    if class_name not in class_map:
                        print(f"Warning: Class '{class_name}' not in provided master class_map. Skipping annotation.")
                        continue
                    
                    yolo_class_id = class_map[class_name]

                    box_w = ann['width']
                    box_h = ann['height']
                    box_l = ann['left']
                    box_t = ann['top']

                    x_center = (box_l + box_w / 2) / img_width
                    y_center = (box_t + box_h / 2) / img_height
                    w_norm = box_w / img_width
                    h_norm = box_h / img_height
                    
                    yolo_annotations.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

                if yolo_annotations:
                    all_annotations.append({
                        "image_path": image_path,
                        "annotations": yolo_annotations,
                        "filename": image_filename
                    })

    print(f"Found {len(all_annotations)} valid annotated images.")
    if not all_annotations:
        print("No data to process. Exiting.")
        return
        
    random.shuffle(all_annotations)
    split_index = int(len(all_annotations) * train_split)
    train_data = all_annotations[:split_index]
    val_data = all_annotations[split_index:]

    print(f"Splitting into {len(train_data)} training and {len(val_data)} validation samples.")

    def process_split(dataset, split_name):
        print(f"Processing {split_name} data...")
        for item in tqdm(dataset):
            shutil.copy(item['image_path'], dirs[f'{split_name}_images'])
            
            label_filename = os.path.splitext(item['filename'])[0] + '.txt'
            label_path = os.path.join(dirs[f'{split_name}_labels'], label_filename)
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(item['annotations']))

    process_split(train_data, 'train')
    process_split(val_data, 'val')

    yaml_path = os.path.join(output_dir, 'data.yaml')
    print(f"Creating dataset YAML file at: {yaml_path}")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write("names:\n")
        sorted_class_names = sorted(class_map.items(), key=lambda item: item[1])
        for name, index in sorted_class_names:
            f.write(f"  {index}: {name}\n")
            
    print("Dataset preparation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert SageMaker GT manifest to YOLOv8 format.")
    parser.add_argument('--manifests', nargs='+', required=True, help="List of paths to manifest files.")
    parser.add_argument('--image_dirs', nargs='+', required=True, help="List of paths to corresponding image directories.")
    parser.add_argument('--output_dir', required=True, help="Directory to save the YOLOv8 dataset.")
    
    args = parser.parse_args()

    CLASS_MAP = {
        'header': 0,
        'passage': 1,
        'question_block': 2,
        'question_number': 3,
        'figure': 4,
        'footer': 5
    }

    if len(args.manifests) != len(args.image_dirs):
        raise ValueError("The number of manifest files must equal the number of image directories.")

    convert_sagemaker_to_yolo(
        manifest_files=args.manifests,
        image_dirs=args.image_dirs,
        output_dir=args.output_dir,
        class_map=CLASS_MAP,
        train_split=0.85
    )