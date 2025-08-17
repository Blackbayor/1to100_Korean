import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import os
import json
from typing import List, Dict, Any

Component = Dict[str, Any]
LogicalUnit = List[Component]

PALETTE = {
    "header": (0, 0, 255),
    "passage": (0, 150, 0),
    "question_block": (220, 0, 0),
    "attachment": (120, 0, 200),
    "other": (120, 120, 120),
}

def recombine_pdf(
    output_pdf_path: str,
    logical_units_to_place: List[LogicalUnit],
    cfg: Dict[str, Any]
):
    """
    재조합 + 항상 배치 맵(JSON) + placement_debug PNGs 생성.
    """
    placement_map = {"pages": [], "output_pdf": os.path.abspath(output_pdf_path)}

    doc = fitz.open()
    page = doc.new_page(width=cfg['page_size'][0], height=cfg['page_size'][1])

    page_width, page_height = cfg['page_size']
    margin = cfg['margin']
    spacing = cfg['spacing_between_components']

    header_y = cfg.get('header_y_position', 70)
    header_line_width = cfg.get('header_line_width', 0.5)
    content_start_y = header_y + spacing

    two_column_layout = cfg.get('two_column_layout', False)
    column_line_width = cfg.get('column_line_width', 0)

    def draw_header(p):
        p.draw_line(fitz.Point(margin, header_y), fitz.Point(page_width - margin, header_y), color=(0, 0, 0), width=header_line_width)

    def _ensure_page_entry(pg_obj):
        pid = int(pg_obj.number)
        if len(placement_map["pages"]) == 0 or placement_map["pages"][-1]["page_id"] != pid:
            placement_map["pages"].append({"page_id": pid, "items": []})

    draw_header(page)
    _ensure_page_entry(page)

    if two_column_layout:
        column_width = (page_width - 3 * margin) / 2
        column_x_pos = [margin, margin + column_width + margin]
        y_cursors = [content_start_y, content_start_y]
    else:
        column_width = page_width - 2 * margin
        column_x_pos = [float(margin)]
        y_cursors = [content_start_y]

    current_column = 0
    current_question_num = cfg.get('start_question_number', 1)

    def get_scaled_dimensions(img_w, img_h):
        max_allowed_width = column_width
        if img_w > max_allowed_width:
            scale = max_allowed_width / img_w
            return int(img_w * scale), int(img_h * scale)
        return int(img_w), int(img_h)

    def ensure_space(h_needed):
        nonlocal page, current_column, y_cursors
        if y_cursors[current_column] + h_needed > page_height - margin:
            if two_column_layout and current_column == 0:
                current_column = 1
            else:
                page = doc.new_page(width=page_width, height=page_height)
                draw_header(page)
                _ensure_page_entry(page)
                y_cursors = [content_start_y, content_start_y] if two_column_layout else [content_start_y]
                current_column = 0

    for unit_idx, unit in enumerate(logical_units_to_place):
        for i, component in enumerate(unit):
            image_path = component['image_path']
            if not os.path.exists(image_path):
                print(f"경고: 이미지를 찾을 수 없습니다. 건너뜁니다 -> {image_path}")
                continue

            with Image.open(image_path) as img:
                w0, h0 = img.size
            w, h = get_scaled_dimensions(w0, h0)

            is_header_followed_by_passage = (
                component['label'] == 'header' and (i + 1) < len(unit) and unit[i + 1]['label'] == 'passage'
            )
            required_height = h
            if is_header_followed_by_passage:
                next_comp = unit[i + 1]
                if os.path.exists(next_comp['image_path']):
                    with Image.open(next_comp['image_path']) as nimg:
                        nw, nh = get_scaled_dimensions(*nimg.size)
                    required_height += spacing + nh

            attachments = component.get('attachments', [])
            for att in attachments:
                if os.path.exists(att['image_path']):
                    with Image.open(att['image_path']) as aimg:
                        aw, ah = get_scaled_dimensions(*aimg.size)
                    required_height += spacing + ah

            ensure_space(required_height)

            x_pos = column_x_pos[current_column]
            y_pos = y_cursors[current_column]

            rect = fitz.Rect(x_pos, y_pos, x_pos + w, y_pos + h)
            page.insert_image(rect, filename=image_path)

            item = {
                "type": component['label'],
                "image_path": image_path,
                "page": int(page.number),
                "column": current_column,
                "x": float(x_pos),
                "y": float(y_pos),
                "w": float(w),
                "h": float(h)
            }

            if component['label'] == "question_block":
                # q_num_text = f"{current_question_num}."
                # text_pos = fitz.Point(x_pos + cfg.get('question_number_offset_x', 10),
                #                       y_pos + cfg.get('question_number_offset_y', 12))
                # page.insert_text(text_pos, q_num_text, fontsize=cfg.get('question_number_font_size', 12), color=(0, 0, 0))
                # item["question_number"] = current_question_num
                # current_question_num += 1
                pass

            y_cursors[current_column] += h + spacing
            _ensure_page_entry(page)
            placement_map["pages"][-1]["items"].append(item)

            for att in attachments:
                apath = att['image_path']
                if not os.path.exists(apath):
                    continue
                with Image.open(apath) as aimg:
                    aw0, ah0 = aimg.size
                aw, ah = get_scaled_dimensions(aw0, ah0)
                ensure_space(ah)
                ax = column_x_pos[current_column]
                ay = y_cursors[current_column]
                rect_att = fitz.Rect(ax, ay, ax + aw, ay + ah)
                page.insert_image(rect_att, filename=apath)
                y_cursors[current_column] += ah + spacing
                _ensure_page_entry(page)
                placement_map["pages"][-1]["items"].append({
                    "type": "attachment",
                    "image_path": apath,
                    "page": int(page.number),
                    "column": current_column,
                    "x": float(ax),
                    "y": float(ay),
                    "w": float(aw),
                    "h": float(ah)
                })

    # divider
    if two_column_layout and column_line_width > 0:
        for p in doc:
            center_x = page_width / 2
            p.draw_line(fitz.Point(center_x, content_start_y),
                        fitz.Point(center_x, page_height - margin),
                        color=(0, 0, 0), width=column_line_width)

    # save pdf + placement
    doc.save(output_pdf_path)
    doc.close()
    json_path = os.path.splitext(output_pdf_path)[0] + "_placement.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(placement_map, f, ensure_ascii=False, indent=2)
    print(f"\nPDF 재조합 완료: {output_pdf_path}")
    print(f"배치 맵 JSON: {json_path}")

    # --- placement debug PNGs ---
    dbg_dir = os.path.splitext(output_pdf_path)[0] + "_placement_debug"
    os.makedirs(dbg_dir, exist_ok=True)
    for page_entry in placement_map["pages"]:
        pid = page_entry["page_id"]
        canvas = Image.new("RGB", (int(page_width), int(page_height)), (255,255,255))
        draw = ImageDraw.Draw(canvas)
        for it in page_entry["items"]:
            color = PALETTE.get(it["type"], PALETTE["other"])
            x, y, w, h = it["x"], it["y"], it["w"], it["h"]
            draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
            label = it["type"]
            if "question_number" in it:
                label += f" #{it['question_number']}"
            draw.text((x+4, max(0, y-14)), label, fill=color)
        out_img = os.path.join(dbg_dir, f"page_{pid:03d}_placement.png")
        canvas.save(out_img)
    print(f"배치 디버그 이미지 폴더: {dbg_dir}")