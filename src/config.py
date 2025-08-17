import os

class Config:
    def __init__(self):
        # --- Project Root ---
        self.PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # --- Data Paths ---
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, 'data')
        self.RAW_DATA_DIR = os.path.join(self.DATA_DIR, 'raw')
        self.PROCESSED_DATA_DIR = os.path.join(self.DATA_DIR, 'processed')
        self.LABELED_DATA_DIR = os.path.join(self.DATA_DIR, 'labeled')
        self.TRAINING_DATA_DIR = os.path.join(self.PROCESSED_DATA_DIR, 'training_data')
        self.IMAGE_DIR = os.path.join(self.PROCESSED_DATA_DIR, 'images')
        self.SAMPLE_ANNOTATIONS_PATH = os.path.join(self.PROCESSED_DATA_DIR, 'sample_annotations.json')
        self.CROPPED_COMPONENTS_DIR = os.path.join(self.PROCESSED_DATA_DIR, 'cropped_components')
        self.INFERENCE_RESULTS_DIR = os.path.join(self.PROCESSED_DATA_DIR, 'inference_results')
        self.RECOMBINED_PDF_OUTPUT_PATH = os.path.join(self.PROCESSED_DATA_DIR, 'recombined_output.pdf')

        # --- AWS S3 Configuration ---
        self.S3_BUCKET = '1to100-ai-data-jykim-250626'
        self.S3_RAW_IMAGES_PATH = 'raw-images/2_image_outputs/'
        self.S3_LABELED_DATA_PATH = 'sagemaker/ground-truth-labels/'
        self.S3_MODEL_OUTPUT_PATH = 'sagemaker/training-output/'

        # --- Model & Training Configuration ---
        self.MODEL_NAME = 'microsoft/layoutlmv3-base'
        self.YOLO_MODEL_PATH = os.path.join(self.PROJECT_ROOT, 'notebooks', 'runs', 'detect', 'train', 'weights', 'best.pt')

        self.CLASSES = ['header', 'passage', 'question_block', 'question_number', 'figure', 'footer']
        self.ID2LABEL = {k: v for k, v in enumerate(self.CLASSES)}
        self.LABEL2ID = {v: k for k, v in enumerate(self.CLASSES)}
        self.CLASS_MAP = {
            0: "header", 1: "passage", 2: "question_block",
            3: "question_number", 4: "figure", 5: "footer"
        }
        self.CLASS_NAMES = {
            0: 'header',
            1: 'passage',
            2: 'question_block',
            3: 'question_number',
            4: 'figure',
            5: 'footer'
        }

        # Training hyperparameters
        self.MAX_LENGTH = 512
        self.BATCH_SIZE = 8
        self.LEARNING_RATE = 5e-5
        self.NUM_EPOCHS = 5

        # --- Post-processing ---
        self.DPI = 72
        self.PDF_STANDARD_DPI = 72
        self.SCALE_FACTOR = self.DPI / self.PDF_STANDARD_DPI

        self.PAGE_SIZES = {
            "A4": (595, 842),
            "B4": (709, 1001)
        }
        self.DEFAULT_PAGE_WIDTH_PT = self.PAGE_SIZES["B4"][0]
        self.DEFAULT_PAGE_HEIGHT_PT = self.PAGE_SIZES["B4"][1]

        self.HEADER_HEIGHT_MM = 15
        self.FOOTER_HEIGHT_MM = 10
        self.TOP_MARGIN_MM = 20
        self.BOTTOM_MARGIN_MM = 15
        self.GUTTER_MARGIN_MM = 10
        self.COLUMN_COUNT = 2

    def mm_to_pt(self, mm):
        return mm * 2.83465

    @property
    def header_height_pt(self):
        return self.mm_to_pt(self.HEADER_HEIGHT_MM)

    @property
    def footer_height_pt(self):
        return self.mm_to_pt(self.FOOTER_HEIGHT_MM)

    @property
    def top_margin_pt(self):
        return self.mm_to_pt(self.TOP_MARGIN_MM)

    @property
    def bottom_margin_pt(self):
        return self.mm_to_pt(self.BOTTOM_MARGIN_MM)

    @property
    def gutter_margin_pt(self):
        return self.mm_to_pt(self.GUTTER_MARGIN_MM)

    def set_request_id(self, request_id: str):
        self.PROCESSED_DATA_DIR = os.path.join(self.DATA_DIR, 'processed', request_id)
        self.IMAGE_DIR = os.path.join(self.PROCESSED_DATA_DIR, 'images')
        self.SAMPLE_ANNOTATIONS_PATH = os.path.join(self.PROCESSED_DATA_DIR, 'sample_annotations.json')
        self.CROPPED_COMPONENTS_DIR = os.path.join(self.PROCESSED_DATA_DIR, 'cropped_components')
        self.INFERENCE_RESULTS_DIR = os.path.join(self.PROCESSED_DATA_DIR, 'inference_results')
        self.RECOMBINED_PDF_OUTPUT_PATH = os.path.join(self.PROCESSED_DATA_DIR, 'recombined_output.pdf')