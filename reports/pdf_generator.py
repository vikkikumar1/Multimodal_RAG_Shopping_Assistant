from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf(data, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    c.drawString(100, 750, "RAG Evaluation Report")
    # Add content
    c.save()