import io
import os
import uuid
import tempfile
import shutil
import fitz  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import FileResponse
from starlette.background import BackgroundTask
from ultralytics import YOLO

# --- 프로젝트의 다른 모듈에서 기능들을 가져옵니다 ---
from src.layout_organizer import group_components, shuffle_logical_units
from src.image_cropper import crop_and_mask_image
from src.pdf_recombiner import recombine_pdf

# 1. FastAPI 앱과 AI 모델을 준비합니다.
app = FastAPI(title="국어 시험지 분석 AI API")
model = YOLO("models/yolo_runs_best.pt")

# --- PDF 재조합에 필요한 기본 설정 ---
DEFAULT_RECOMBINE_CONFIG = {
    'page_size': (595, 842),  # A4
    'margin': 50,
    'spacing_between_components': 15,
    'header_y_position': 70,
    'header_line_width': 0.5,
    'two_column_layout': False,
    'image_scale_factor': 1.0,
    'start_question_number': 1,
    'question_number_offset_x': 20,
    'question_number_offset_y': 20,
    'question_number_font_size': 10,
}

# --- Helper Function ---
def cleanup_temp_dir(temp_dir: str):
    """임시 디렉터리와 그 안의 모든 파일을 삭제하는 정리 함수"""
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"임시 디렉터리 삭제 실패: {temp_dir}, 오류: {e}")

@app.get("/")
def read_root():
    return {"message": "국어 시험지 분석 API에 오신 것을 환영합니다! /docs 로 접속하여 API 문서를 확인하세요."}

# ... (기존 /predict/ 엔드포인트는 여기에 위치, 생략) ...

@app.post("/recombine/", summary="[최종] PDF 재조합 및 다운로드", description="시험지 PDF를 받아 문제를 섞은 새 PDF를 반환합니다.")
async def recombine_and_generate_pdf(file: UploadFile = File(..., description="재조합할 원본 PDF 파일")):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    contents = await file.read()
    temp_dir = tempfile.mkdtemp()  # 임시 디렉터리 생성

    try:
        pdf_document = fitz.open(stream=contents, filetype="pdf")
        all_logical_units = []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=72)
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))

            results = model(page_image)
            predictions = []
            for r in results:
                for box in r.boxes:
                    predictions.append({
                        "class_name": model.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "coordinates": [round(c) for c in box.xyxy[0].tolist()]
                    })

            logical_units_on_page = group_components(predictions)

            for unit in logical_units_on_page:
                for component in unit:
                    bbox = component['coordinates']
                    cropped_img = crop_and_mask_image(page_image, bbox)
                    temp_img_path = os.path.join(temp_dir, f"{uuid.uuid4()}.png")
                    cropped_img.save(temp_img_path)
                    component['image_path'] = temp_img_path
            
            all_logical_units.extend(logical_units_on_page)

        pdf_document.close()

        shuffled_units = shuffle_logical_units(all_logical_units)
        output_pdf_path = os.path.join(temp_dir, "recombined_output.pdf")
        recombine_pdf(output_pdf_path, shuffled_units, DEFAULT_RECOMBINE_CONFIG)

        # 파일을 전송하고, 전송이 완료된 후에 cleanup_temp_dir 함수를 실행하도록 설정
        return FileResponse(
            path=output_pdf_path,
            media_type='application/pdf',
            filename='shuffled_exam.pdf',
            background=BackgroundTask(cleanup_temp_dir, temp_dir)
        )

    except Exception as e:
        # 파이프라인 실행 중 어떤 단계에서든 오류가 발생하면, 임시 폴더를 정리하고 에러를 반환
        cleanup_temp_dir(temp_dir)
        raise HTTPException(status_code=500, detail=f"PDF 재조합 중 오류 발생: {e}")
