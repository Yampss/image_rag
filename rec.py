from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'   
image = Image.open('istockphoto-1420767944-612x612.jpg')
text = pytesseract.image_to_string( image )
print(text)
from sentence_transformers import SentenceTransformer


# Step 2: Convert the extracted text to vectors
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models as well
# vector = model.encode(text)
vector = model.encode(text).tolist()


print(f"Extracted Text: {text}")
print(f"Text Vector: {vector}")
import os
import pinecone
from pinecone import Pinecone
pc = Pinecone(
    api_key="19ddd4aa-9fbc-44f7-a499-6f194785e9c5"  # Replace with your actual API key
)

index_name = 'image-text-vectors'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=len(vector),
        metric='euclidean',  # You can use other metrics like 'cosine'
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Use a supported region like us-west-2
        )


    )
index = pc.Index(index_name)
index.upsert(vectors=[(image.filename, vector)])


print(f"Vector for {image.filename} stored successfully.")
# Retrieve the vector using the ID (image filename)
retrieved_vector = index.fetch(ids=[image.filename])


# Retrieve the vector using the ID (image filename)
retrieved_vector = index.fetch(ids=[image.filename])
print(f"Retrieved Vector for {image.filename}: {retrieved_vector['vectors'][image.filename]}")



# import ollama
# response = ollama.chat(model='llama3.1', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])