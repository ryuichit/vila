import sys

import layoutparser as lp # For visualization 

from vila.pdftools.pdf_extractor import PDFExtractor
from vila.predictors import HierarchicalPDFPredictor, LayoutIndicatorPDFPredictor
# Choose from SimplePDFPredictor,
# LayoutIndicatorPDFPredictor, 
# and HierarchicalPDFPredictor

PDF_PATH = sys.argv[1]
MAX_PAGE = int(sys.argv[2]) if len(sys.argv) > 2 else 1

pdf_extractor = PDFExtractor("pdfplumber")
page_tokens, page_images = pdf_extractor.load_tokens_and_image(PDF_PATH)

vision_model = lp.EfficientDetLayoutModel("lp://PubLayNet") 
pdf_predictor = LayoutIndicatorPDFPredictor.from_pretrained("allenai/ivila-block-layoutlm-finetuned-docbank")

for idx, page_token in enumerate(page_tokens):
    if idx >= MAX_PAGE:
        break
    blocks = vision_model.detect(page_images[idx])
    page_token.annotate(blocks=blocks)
    pdf_data = page_token.to_pagedata().to_dict()
    predicted_tokens = pdf_predictor.predict(pdf_data, page_token.page_size)
    print(idx, predicted_tokens)
    lp.draw_box(page_images[idx], predicted_tokens, box_width=0, box_alpha=0.25).show()
