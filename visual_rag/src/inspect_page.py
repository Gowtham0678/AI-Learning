from PIL import Image

image_path = "rendered_pages/page_006.png"

image = Image.open(image_path)

print("Image size:", image.size)
print("Image mode:", image.mode)
image.show()