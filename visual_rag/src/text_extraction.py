import pymupdf

pdf = pymupdf.open("mcp.pdf")

page = pdf[5]  # Page 6

text = page.get_text()

print("----- EXTRACTED TEXT -----")
print(text)

pdf.close()