import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --------------------------------------------------
# 1. Load pretrained CLIP
# --------------------------------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

print("CLIP loaded successfully.\n")


# --------------------------------------------------
# 2. Get image path and query
# --------------------------------------------------

image_path = input("Enter screenshot path: ")

query = input("What are you looking for? ")


# --------------------------------------------------
# 3. Load screenshot
# --------------------------------------------------

image = Image.open(image_path).convert("RGB")


# --------------------------------------------------
# 4. Convert image + text into CLIP representations
# --------------------------------------------------

inputs = processor(
    text=[query],
    images=image,
    return_tensors="pt",
    padding=True
)


# --------------------------------------------------
# 5. Run CLIP
# --------------------------------------------------

with torch.no_grad():
    outputs = model(**inputs)

    image_embedding = outputs.image_embeds
    text_embedding = outputs.text_embeds


# --------------------------------------------------
# 6. Normalize embeddings
# --------------------------------------------------

image_embedding = image_embedding / image_embedding.norm(
    dim=-1, keepdim=True
)

text_embedding = text_embedding / text_embedding.norm(
    dim=-1, keepdim=True
)


# --------------------------------------------------
# 7. Calculate cosine similarity
# --------------------------------------------------

similarity = torch.matmul(
    image_embedding,
    text_embedding.T
)

score = similarity.item()


# --------------------------------------------------
# 8. Display result
# --------------------------------------------------

print("\n==============================")
print("       SCREENMIND RESULT")
print("==============================")

print(f"\nQuery: {query}")
print(f"Similarity score: {score:.4f}")

print("\nHigher score = stronger semantic match.")