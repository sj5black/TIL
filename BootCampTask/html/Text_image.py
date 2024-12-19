import pytesseract
from PIL import Image
from googletrans import Translator

# Set path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Modify this path for your system

# Load and display image
image_path = 'path_to_image.jpg'  # Replace with your image path
image = Image.open(image_path)

# Use pytesseract to extract text from image
extracted_text = pytesseract.image_to_string(image, lang='eng')  # You can change the language model

print("Extracted Text:")
print(extracted_text)

# Initialize translator
translator = Translator()

# Translate text to the desired language (e.g., Korean 'ko')
translated_text = translator.translate(extracted_text, dest='ko')

print("\nTranslated Text:")
print(translated_text.text)
