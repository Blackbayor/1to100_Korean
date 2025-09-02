import json
import os

# --- 설정 (자동으로 구성됨) ---
# 1. SageMaker에서 다운로드한 manifest 파일 경로
sagemaker_manifest_file = r'C:\Users\CUBOX\1to100\aws\2_image_outputs\suneung-korean-layout-detection-v2\manifests\output\output.manifest'

# 2. 변환된 Label Studio용 json 파일을 저장할 경로
label_studio_output_file = r'C:\Users\CUBOX\1to100\aws\manifest_to_json\label_studio_import.json'

# 3. 이미지 파일들이 로컬 컴퓨터에 저장된 경로
local_image_base_path = r"C:\Users\CUBOX\1to100\aws\2_image_outputs\rawdata"
# --- 설정 끝 ---


tasks = []
with open(sagemaker_manifest_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)

        # 이미지 파일 경로 추출 및 로컬 경로로 변환
        s3_path = data['source-ref']
        image_filename = os.path.basename(s3_path)
        
        # f-string 외부에서 경로 구분자 처리
        sanitized_path = local_image_base_path.replace('\\', '/')
        image_url_for_ls = f"/data/local-files/?d={sanitized_path}/{image_filename}"

        # 라벨링 결과 추출 (작업 이름은 실제 이름에 맞게 조정 필요)
        job_name = None
        for key in data:
            if not key.endswith('-metadata') and key != 'source-ref':
                job_name = key
                break
        
        if not job_name or job_name not in data or 'annotations' not in data[job_name]:
            continue
        
        metadata_key = f"{job_name}-metadata"
        if metadata_key not in data or 'class-map' not in data[metadata_key]:
            continue

        # 클래스 ID와 라벨 이름 매핑 정보 추출
        class_map = data[metadata_key]['class-map']
            
        annotations = data[job_name]['annotations']
        image_width = data[job_name]['image_size'][0]['width']
        image_height = data[job_name]['image_size'][0]['height']

        # Label Studio 형식으로 변환
        results = []
        for ann in annotations:
            class_id = str(ann['class_id']) # class_map의 키는 문자열
            label_name = class_map.get(class_id, "unknown") # ID에 해당하는 라벨 이름 찾기

            # SageMaker 좌표 (top, left, height, width) -> Label Studio 좌표 (x, y, width, height in %)
            x_coord = (ann['left'] / image_width) * 100
            y_coord = (ann['top'] / image_height) * 100
            width_perc = (ann['width'] / image_width) * 100
            height_perc = (ann['height'] / image_height) * 100

            results.append({
                "from_name": "label", # Label Studio 설정과 일치해야 함
                "to_name": "image",   # Label Studio 설정과 일치해야 함
                "type": "rectanglelabels",
                "original_width": image_width,
                "original_height": image_height,
                "image_rotation": 0,
                "value": {
                    "x": x_coord,
                    "y": y_coord,
                    "width": width_perc,
                    "height": height_perc,
                    "rotation": 0,
                    "rectanglelabels": [label_name]
                }
            })
        
        tasks.append({
            "data": {"image": image_url_for_ls},
            "annotations": [{"result": results}]
        })

# 변환된 데이터를 JSON 파일로 저장
with open(label_studio_output_file, 'w', encoding='utf-8') as f:
    json.dump(tasks, f, ensure_ascii=False, indent=4)

print(f"'{label_studio_output_file}' 파일이 생성되었습니다. Label Studio에서 Import 하세요.")
