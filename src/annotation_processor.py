import os
import json
from typing import List, Dict, Tuple, Any, Optional
from PIL import Image, ImageDraw
from collections import defaultdict

from .image_cropper import crop_and_mask_image, Bbox
from .config import Config

# --- Type Aliases ---
Component = Dict[str, Any]
LogicalUnit = List[Component]

PALETTE = {
    "header": (0, 0, 255),
    "passage": (0, 150, 0),
    "question_block": (220, 0, 0),
    "question_number": (200, 120, 0),
    "figure": (120, 0, 200),
    "footer": (120, 120, 120),
}

def _area(b):
    return max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))

def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    if inter <= 0: return 0.0
    ua = _area(a) + _area(b) - inter
    return inter / ua if ua > 0 else 0.0

def _nms(annos: List[Dict[str, Any]], iou_thr: float) -> List[Dict[str, Any]]:
    annos_sorted = sorted(annos, key=lambda d: d.get('confidence', 0.5), reverse=True)
    kept = []
    for a in annos_sorted:
        if all(_iou(a['bbox'], b['bbox']) < iou_thr for b in kept):
            kept.append(a)
    return kept

def _kmeans_two_columns(x_centers: List[float]):
    try:
        from sklearn.cluster import KMeans
        import numpy as np
        X = np.array(x_centers).reshape(-1,1)
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
        centers = sorted([c[0] for c in km.cluster_centers_])
        threshold = sum(centers)/2.0
        return threshold, centers[0], True
    except Exception:
        if not x_centers: return None
        xs = sorted(x_centers)
        mid = xs[len(xs)//2]
        return mid, xs[0], False

def _point_in_bbox(pt, bbox):
    x,y = pt
    return (bbox[0] <= x <= bbox[2]) and (bbox[1] <= y <= bbox[3])

def _center(b):
    return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

def _distance(p, q):
    return ((p[0]-q[0])**2 + (p[1]-q[1])**2) ** 0.5

def _merge_adjacent_blocks(blocks: List[Dict[str, Any]], x_overlap_ratio=0.6, max_vgap_px=80,
                           merge_trace: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    merged = []
    for b in blocks:
        if not merged:
            b['_merge_id'] = len(merged)
            merged.append(b); continue
        last = merged[-1]
        left = max(last['bbox'][0], b['bbox'][0])
        right = min(last['bbox'][2], b['bbox'][2])
        overlap_w = max(0.0, right - left)
        width_ref = max(last['bbox'][2]-last['bbox'][0], 1.0)
        vgap = b['bbox'][1] - last['bbox'][3]
        decision = (overlap_w / width_ref >= x_overlap_ratio) and (0 <= vgap <= max_vgap_px)
        if decision:
            new_bbox = (
                min(last['bbox'][0], b['bbox'][0]),
                min(last['bbox'][1], b['bbox'][1]),
                max(last['bbox'][2], b['bbox'][2]),
                max(last['bbox'][3], b['bbox'][3]),
            )
            if merge_trace is not None:
                merge_trace.append({
                    "merge": {"from": last.get('_merge_id', -1), "with": len(merged)},
                    "x_overlap_ratio": overlap_w/width_ref if width_ref else 0.0,
                    "vgap": vgap
                })
            children = last.get('children', []) + b.get('children', [])
            attachments = last.get('attachments', []) + b.get('attachments', [])
            last.update({'bbox': new_bbox, 'children': children, 'attachments': attachments})
        else:
            b['_merge_id'] = len(merged)
            merged.append(b)
    return merged

def _draw_boxes(image_path: str, annos: List[Dict[str, Any]], outfile: str, title: Optional[str] = None):
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception:
        return
    draw = ImageDraw.Draw(im)
    for idx, a in enumerate(annos):
        bbox = a["bbox"]
        label = a["label"]
        color = PALETTE.get(label, (0,0,0))
        draw.rectangle(bbox, outline=color, width=2)
        txt = f"{label[:10]}#{idx}"
        tag_h = 16
        draw.rectangle((bbox[0], max(0, bbox[1]-tag_h), bbox[0]+8*len(txt), bbox[1]), fill=color)
    if title:
        draw.text((10, 10), title, fill=(255,255,255))
    im.save(outfile)

def process_annotations_from_json(json_file_path: str, base_output_dir: str, config: Config) -> List[LogicalUnit]:
    """
    디버그 산출물 + 의사결정 근거를 JSON으로 남깁니다.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        pages = json.load(f)

    os.makedirs(base_output_dir, exist_ok=True)

    processed_dir = getattr(config, "PROCESSED_DATA_DIR", None)
    if processed_dir:
        debug_root = os.path.join(processed_dir, "debug")
    else:
        debug_root = os.path.normpath(os.path.join(base_output_dir, "..", "debug"))
    os.makedirs(debug_root, exist_ok=True)

    global_report = {
        "pages": [],
        "totals": {"input_boxes": 0, "after_filter": 0, "question_numbers_attached": 0, "question_numbers_orphan": 0,
                   "figures_attached": 0, "logical_units": 0},
        "paths": {"debug_dir": os.path.abspath(debug_root), "cropped_dir": os.path.abspath(base_output_dir)}
    }

    processed_pages: List[List[Dict[str, Any]]] = []

    for page_index, page_data in enumerate(pages):
        image_path = page_data["image_path"]
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception:
            img_w = 2000; img_h = 3000

        page_report = {
            "page_index": page_index,
            "image_path": image_path,
            "input_count": len(page_data["annotations"]),
            "after_filter_count": 0,
            "kmeans_used": False,
            "column_threshold": None,
            "merged_qb_count": 0,
            "numbers_attached": 0,
            "numbers_orphan": 0,
            "figures_attached": 0,
            "label_counts": {},
            "column_assignment": [],
            "number_mapping_trace": [],
            "merge_trace": []
        }

        # 1) filter
        raw = []
        for a in page_data["annotations"]:
            bbox = tuple(map(float, a["bbox"]))
            w = max(0.0, bbox[2]-bbox[0]); h = max(0.0, bbox[3]-bbox[1])
            area_ratio = (w*h) / max(1.0, img_w*img_h)
            conf = float(a.get("confidence", 0.5))
            label = a["label"]
            min_conf = {"question_number":0.40, "figure":0.50}.get(label, 0.35)
            if conf < min_conf: 
                continue
            if area_ratio < 0.002:
                continue
            if label == "question_number":
                ar = (max(w,h)/max(1.0, min(w,h)))
                if ar > 5.0:
                    continue
            a = dict(a)
            a.update({"bbox": bbox, "page_index": page_index, "original_image_path": image_path})
            raw.append(a)

        # 2) NMS per class
        by_cls = defaultdict(list)
        for a in raw:
            by_cls[a["label"]].append(a)
        filtered = []
        for lbl, arr in by_cls.items():
            keep = _nms(arr, iou_thr=0.5)
            filtered.extend(keep)
            page_report["label_counts"][lbl] = len(keep)

        # 3) merge split qbs with trace
        qbs = [a for a in filtered if a["label"]=="question_block"]
        others = [a for a in filtered if a["label"]!="question_block"]
        before = len(qbs)
        qbs = sorted(qbs, key=lambda x: (x["bbox"][1], x["bbox"][0]))
        qbs = _merge_adjacent_blocks(qbs, x_overlap_ratio=0.6, max_vgap_px=int(img_h*0.03), merge_trace=page_report["merge_trace"])
        page_report["merged_qb_count"] = before - len(qbs)
        filtered = qbs + others

        # 4) columns
        x_centers = [ (a["bbox"][0]+a["bbox"][2])/2.0 for a in filtered if a["label"]!="footer" ]
        res = _kmeans_two_columns(x_centers) if x_centers else None
        if res:
            threshold_x, _, used = res
            page_report["kmeans_used"] = used
            page_report["column_threshold"] = threshold_x
        else:
            threshold_x = img_w/2.0
            page_report["kmeans_used"] = False
            page_report["column_threshold"] = threshold_x

        for idx, a in enumerate(filtered):
            xc = (a["bbox"][0]+a["bbox"][2])/2.0
            a["column"] = 0 if xc < threshold_x else 1
            page_report["column_assignment"].append({"idx": idx, "label": a["label"], "xc": xc, "column": a["column"]})

        # 5) number→block with trace
        blocks = [a for a in filtered if a["label"]=="question_block"]
        numbers = [a for a in filtered if a["label"]=="question_number"]
        for i, b in enumerate(blocks):
            b["children"] = []
            b["attachments"] = []
            b["_tmp_block_id"] = i
        attached = 0
        for qn in numbers:
            qn_center = _center(qn["bbox"])
            method = "none"
            candidate = None
            for b in blocks:
                if _point_in_bbox(qn_center, b["bbox"]):
                    candidate = b; method = "containment"; break
            dist_val = None
            if candidate is None:
                same_col = [b for b in blocks if b["column"]==qn["column"]]
                cands = same_col if same_col else blocks
                if cands:
                    scored = [(b, _distance(_center(b["bbox"]), qn_center)) for b in cands]
                    scored.sort(key=lambda x: x[1])
                    if scored and scored[0][1] <= max(img_h*0.1, 120):
                        candidate = scored[0][0]
                        method = "nearest"
                        dist_val = scored[0][1]
            if candidate:
                candidate["children"].append(qn)
                attached += 1
                page_report["number_mapping_trace"].append({
                    "qn_bbox": list(map(float, qn["bbox"])),
                    "mapped_block_id": candidate["_tmp_block_id"],
                    "method": method,
                    "distance": dist_val
                })
            else:
                page_report["number_mapping_trace"].append({
                    "qn_bbox": list(map(float, qn["bbox"])),
                    "mapped_block_id": None,
                    "method": "orphan",
                    "distance": None
                })
        page_report["numbers_attached"] = attached
        page_report["numbers_orphan"] = max(0, len(numbers) - attached)

        # 6) figures attach
        figures = [a for a in filtered if a["label"]=="figure"]
        passages = [a for a in filtered if a["label"]=="passage"]
        f_attached = 0
        for fig in figures:
            fig_center = _center(fig["bbox"])
            cands = blocks + passages
            if not cands: 
                continue
            dists = [(c, _distance(_center(c["bbox"]), fig_center)) for c in cands]
            dists.sort(key=lambda x: x[1])
            host = dists[0][0]
            host.setdefault("attachments", []).append(fig)
            f_attached += 1
        page_report["figures_attached"] = f_attached

        # sort + save overlay
        filtered_sorted = sorted(filtered, key=lambda x: (x["column"], x["bbox"][1], x["bbox"][0]))
        processed_pages.append(filtered_sorted)
        page_report["after_filter_count"] = len(filtered_sorted)
        global_report["pages"].append(page_report)

        overlay_path = os.path.join(debug_root, f"page_{page_index:03d}_filtered.png")
        _draw_boxes(image_path, filtered_sorted, overlay_path, title=f"page {page_index}")

    # crop + logical units
    def _crop_component(anno: Dict[str, Any], mask_children: Optional[List[Dict[str, Any]]] = None) -> str:
        label = anno['label']
        bbox = tuple(int(round(c)) for c in anno['bbox'])
        image_path = anno['original_image_path']

        original_image = Image.open(image_path)
        cropped_image = crop_and_mask_image(original_image, bbox)

        if label == "question_block" and mask_children:
            rel_masks = []
            for child in mask_children:
                cb = child['bbox']
                rel = (int(round(cb[0]-bbox[0])), int(round(cb[1]-bbox[1])),
                       int(round(cb[2]-bbox[0])), int(round(cb[3]-bbox[1])))
                rel_masks.append(rel)
            if rel_masks:
                draw = ImageDraw.Draw(cropped_image)
                for mb in rel_masks:
                    draw.rectangle(mb, fill="white")

        label_dir = os.path.join(base_output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        idx = len(os.listdir(label_dir))
        out_path = os.path.join(label_dir, f"{base}_{label}_{idx}.png")
        cropped_image.save(out_path)
        return out_path

    logical_units: List[LogicalUnit] = []
    current_unit: LogicalUnit = []
    all_annos = [a for page in processed_pages for a in page]

    for anno in all_annos:
        label = anno["label"]
        if label in ("figure", "question_number", "footer"):
            continue

        start_new = False
        if label == "header":
            start_new = True
        elif label == "passage":
            if not current_unit or (current_unit and current_unit[-1]['label'] == "question_block"):
                start_new = True
        elif label == "question_block":
            if not current_unit or (current_unit and current_unit[-1]['label'] not in ["passage", "question_block"]):
                start_new = True

        if start_new and current_unit:
            logical_units.append(current_unit)
            current_unit = []

        if label == "question_block":
            img_path = _crop_component(anno, mask_children=anno.get("children", []))
            comp: Component = {"label": "question_block", "image_path": img_path, "text_content": ""}
            atts = []
            for fig in anno.get("attachments", []):
                att_path = _crop_component(fig, mask_children=None)
                atts.append({"label": "figure", "image_path": att_path})
            if atts:
                comp["attachments"] = atts
            current_unit.append(comp)
        else:
            img_path = _crop_component(anno)
            comp = {"label": label, "image_path": img_path, "text_content": ""}
            atts = []
            for fig in anno.get("attachments", []):
                att_path = _crop_component(fig, mask_children=None)
                atts.append({"label": "figure", "image_path": att_path})
            if atts:
                comp["attachments"] = atts
            current_unit.append(comp)

    if current_unit:
        logical_units.append(current_unit)

    # finalize report
    global_report["totals"]["input_boxes"] = sum(p["input_count"] for p in global_report["pages"])
    global_report["totals"]["after_filter"] = sum(p["after_filter_count"] for p in global_report["pages"])
    global_report["totals"]["question_numbers_attached"] = sum(p["numbers_attached"] for p in global_report["pages"])
    global_report["totals"]["question_numbers_orphan"] = sum(p["numbers_orphan"] for p in global_report["pages"])
    global_report["totals"]["figures_attached"] = sum(p["figures_attached"] for p in global_report["pages"])
    global_report["totals"]["logical_units"] = len(logical_units)

    with open(os.path.join(debug_root, "annotation_debug_report.json"), "w", encoding="utf-8") as f:
        json.dump(global_report, f, ensure_ascii=False, indent=2)

    light_units = []
    for u in logical_units:
        light_u = []
        for c in u:
            item = {"label": c["label"], "image_path": c["image_path"]}
            if "attachments" in c:
                item["attachments"] = [{"label": a["label"], "image_path": a["image_path"]} for a in c["attachments"]]
            light_u.append(item)
        light_units.append(light_u)
    with open(os.path.join(debug_root, "logical_units.json"), "w", encoding="utf-8") as f:
        json.dump(light_units, f, ensure_ascii=False, indent=2)

    print(f"[annotation_processor] Debug dir: {os.path.abspath(debug_root)}")
    print(f"[annotation_processor] Cropped dir: {os.path.abspath(base_output_dir)}")
    return logical_units