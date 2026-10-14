import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

image_path = input("Enter screenshot path: ")

image = Image.open(image_path)

text = pytesseract.image_to_string(image)

print("\n===== OCR RESULT =====\n")
print(text)