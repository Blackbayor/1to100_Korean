import fitz  # PyMuPDF
from PIL import Image
import os
from typing import List, Dict, Any

# --- Type Aliases for Clarity ---
Component = Dict[str, Any]
LogicalUnit = List[Component]

def recombine_pdf(
    output_pdf_path: str,
    logical_units_to_place: List[LogicalUnit],
    cfg: Dict[str, Any]
):
    """
    논리적 단위에 따라 컴포넌트 이미지들을 새 PDF 페이지에 재조합합니다.
    - ★★★ 모든 페이지 상단에 머리글(가로선)을 추가하는 기능이 포함되었습니다. ★★★
    """
    doc = fitz.open()
    page = doc.new_page(width=cfg['page_size'][0], height=cfg['page_size'][1])
    
    page_width, page_height = cfg['page_size']
    margin = cfg['margin']
    spacing = cfg['spacing_between_components']
    
    # --- 머리글 및 레이아웃 설정 ---
    header_y = cfg.get('header_y_position', 70)  # 머리글 선의 Y좌표
    header_line_width = cfg.get('header_line_width', 0.5)
    content_start_y = header_y + spacing # 머리글 아래 콘텐츠가 시작될 Y좌표

    two_column_layout = cfg.get('two_column_layout', False)
    column_line_width = cfg.get('column_line_width', 0)
    
    # ★★★ 머리글을 그리는 헬퍼 함수 정의 ★★★
    def draw_header(p: fitz.Page):
        p.draw_line(
            fitz.Point(margin, header_y),
            fitz.Point(page_width - margin, header_y),
            color=(0, 0, 0),
            width=header_line_width
        )

    # --- 초기 페이지 설정 ---
    draw_header(page) # 첫 페이지에 머리글 그리기
    
    if two_column_layout:
        column_width = (page_width - 3 * margin) / 2
        column_x_pos = [margin, margin + column_width + margin]
        y_cursors = [content_start_y, content_start_y] # ★★★ 시작 위치 변경
    else:
        column_width = page_width - 2 * margin
        column_x_pos = [float(margin)]
        y_cursors = [content_start_y] # ★★★ 시작 위치 변경
        
    current_column = 0
    current_question_num = cfg['start_question_number']

    for unit_idx, unit in enumerate(logical_units_to_place):
        for i, component in enumerate(unit):
            image_path = component['image_path']
            if not os.path.exists(image_path):
                print(f"경고: 이미지를 찾을 수 없습니다. 건너뜁니다 -> {image_path}")
                continue

            with Image.open(image_path) as img:
                original_img_width, original_img_height = img.size

            scale_factor_from_config = cfg.get('image_scale_factor', 1.0)
            
            # Calculate scaled image dimensions
            def get_scaled_dimensions(img_w, img_h):
                desired_img_width = int(img_w * scale_factor_from_config)
                desired_img_height = int(img_h * scale_factor_from_config)
                
                max_allowed_width = column_width
                if desired_img_width > max_allowed_width:
                    final_scale = max_allowed_width / desired_img_width
                    return int(desired_img_width * final_scale), int(desired_img_height * final_scale)
                return desired_img_width, desired_img_height

            img_width, img_height = get_scaled_dimensions(original_img_width, original_img_height)

            # --- 페이지/컬럼 전환 로직 (Header-Passage 결합 고려) ---
            
            # 1. 현재 컴포넌트가 header이고, 다음에 passage가 오는지 확인
            is_header_followed_by_passage = False
            if component['class_name'] == 'header' and (i + 1) < len(unit) and unit[i + 1]['class_name'] == 'passage':
                is_header_followed_by_passage = True

            # 2. 공간 확인
            required_height = img_height
            if is_header_followed_by_passage:
                # passage 이미지의 예상 높이 계산
                next_comp = unit[i + 1]
                with Image.open(next_comp['image_path']) as next_img:
                    next_img_w, next_img_h = next_img.size
                _, next_img_height_scaled = get_scaled_dimensions(next_img_w, next_img_h)
                required_height += spacing + next_img_height_scaled

            # 3. 공간이 부족하면 페이지/컬럼 전환
            if y_cursors[current_column] + required_height > page_height - margin:
                if two_column_layout and current_column == 0:
                    current_column = 1
                else:
                    page = doc.new_page(width=page_width, height=page_height)
                    draw_header(page)
                    y_cursors = [content_start_y, content_start_y] if two_column_layout else [content_start_y]
                    current_column = 0

            # --- 이미지 삽입 및 y커서 업데이트 ---
            x_pos = column_x_pos[current_column]
            current_y = y_cursors[current_column]
            rect = fitz.Rect(x_pos, current_y, x_pos + img_width, current_y + img_height)
            page.insert_image(rect, filename=image_path)
            
            if component['class_name'] == "question_block":
                q_num_text = f"{current_question_num}."
                text_pos = fitz.Point(
                    x_pos + cfg['question_number_offset_x'],
                    current_y + cfg['question_number_offset_y']
                )
                page.insert_text(
                    text_pos, q_num_text,
                    fontsize=cfg['question_number_font_size'], color=(0, 0, 0)
                )
                current_question_num += 1
            
            y_cursors[current_column] += img_height + spacing

    # 2단 레이아웃일 경우 모든 페이지에 세로 구분선 그리기
    if two_column_layout and column_line_width > 0:
        for p in doc:
            center_x = page_width / 2
            p.draw_line(fitz.Point(center_x, content_start_y), fitz.Point(center_x, page_height - margin),
                        color=(0, 0, 0), width=column_line_width)

    doc.save(output_pdf_path)
    doc.close()
    print(f"\nPDF 재조합 완료: {output_pdf_path}")