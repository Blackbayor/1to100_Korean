import json

# --- 설정 (사용자 수정 필요) ---
# 1. SageMaker에서 다운로드한 manifest 파일 경로
sagemaker_manifest_file = 'C:\\Users\\CUBOX\\1to100\\aws\\2_image_outputs\\suneung-korean-layout-detection-v2\\manifests\\output\\output.manifest'

# 2. 변환된 Label Studio용 json 파일을 저장할 경로
label_studio_output_file = 'C:\\Users\\CUBOX\\1to100\\aws\\manifest_to_json\\label_studio_import.json'

# 3. 이미지 파일들이 로컬 컴퓨터에 저장된 경로
# Label Studio가 이미지를 불러올 수 있도록 웹 경로 또는 로컬 경로를 지정해야 합니다.
# 예: "images/2024_suneung_page_00.png" 와 같이 만들기 위한 기본 경로
local_image_base_path = "C:\\Users\\CUBOX\\1to100\\aws\\2_image_outputs\\suneung-korean-layout-detection-v2" 
# --- 설정 끝 ---


tasks = []
with open(sagemaker_manifest_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)

        # 이미지 파일 경로 추출 및 로컬 경로로 변환
        s3_path = data['source-ref']
        image_filename = s3_path.split('/')[-1]
        
        # Label Studio에서 접근할 수 있는 이미지 경로 생성
        # 여기서는 상대 경로로 지정합니다. Label Studio 설정에 따라 변경될 수 있습니다.
        image_url_for_ls = f"/data/local-files/?d={local_image_base_path}/{image_filename}"

        # 라벨링 결과 추출 (작업 이름은 실제 이름에 맞게 조정 필요)
        job_name = None
        for key in data:
            if not key.endswith('-metadata') and key != 'source-ref':
                job_name = key
                break
        
        if not job_name:
            continue
            
        annotations = data[job_name]['annotations']
        image_width = data[job_name]['image_size'][0]['width']
        image_height = data[job_name]['image_size'][0]['height']

        # Label Studio 형식으로 변환
        results = []
        for ann in annotations:
            # SageMaker 좌표 (top, left, height, width) -> Label Studio 좌표 (x, y, width, height in %)
            x_coord = (ann['left'] / image_width) * 100
            y_coord = (ann['top'] / image_height) * 100
            width_perc = (ann['width'] / image_width) * 100
            height_perc = (ann['height'] / image_height) * 100

            results.append({
                "from_name": "label", # Label Studio 설정과 일치해야 함
                "to_name": "image",   # Label Studio 설정과 일치해야 함
                "type": "rectanglelabels",
                "value": {
                    "x": x_coord,
                    "y": y_coord,
                    "width": width_perc,
                    "height": height_perc,
                    "rectanglelabels": [ann['label']]
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