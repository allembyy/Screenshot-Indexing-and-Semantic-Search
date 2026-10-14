import os
import pickle

import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_FOLDER = "screenshots"
INDEX_FILE = "screenmind.index"
IMAGE_LIST_FILE = "images.pkl"

MODEL_NAME = "openai/clip-vit-base-patch32"


# ==========================================
# LOAD CLIP
# ==========================================

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

print("CLIP loaded successfully.\n")


# ==========================================
# FIND SCREENSHOTS
# ==========================================

extensions = (".png", ".jpg", ".jpeg", ".webp")

image_paths = [
    os.path.join(IMAGE_FOLDER, filename)
    for filename in os.listdir(IMAGE_FOLDER)
    if filename.lower().endswith(extensions)
]

print(f"Found {len(image_paths)} screenshots.\n")

if not image_paths:
    print("No screenshots found.")
    exit()


# ==========================================
# GENERATE IMAGE EMBEDDINGS
# ==========================================

embeddings = []

for i, image_path in enumerate(image_paths):

    print(
        f"Processing {i + 1}/{len(image_paths)}: "
        f"{os.path.basename(image_path)}"
    )

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Prepare image for CLIP
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Generate image features
    with torch.no_grad():

        output = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

        image_embedding = output.pooler_output

    # Normalize embedding
    image_embedding = image_embedding / image_embedding.norm(
        dim=-1,
        keepdim=True
    )

    # Convert to NumPy
    embeddings.append(
        image_embedding.cpu().numpy()[0]
    )


# ==========================================
# CREATE EMBEDDING MATRIX
# ==========================================

embedding_matrix = torch.tensor(
    embeddings,
    dtype=torch.float32
).numpy()

dimension = embedding_matrix.shape[1]


# ==========================================
# CREATE FAISS INDEX
# ==========================================

print("\nCreating FAISS index...")

index = faiss.IndexFlatIP(dimension)

index.add(embedding_matrix)


# ==========================================
# SAVE INDEX
# ==========================================

faiss.write_index(
    index,
    INDEX_FILE
)

with open(IMAGE_LIST_FILE, "wb") as f:
    pickle.dump(
        image_paths,
        f
    )


# ==========================================
# COMPLETE
# ==========================================

print("\n==============================")
print(" SCREENMIND INDEX COMPLETE")
print("==============================")

print(f"Screenshots indexed: {len(image_paths)}")
print(f"Embedding dimension: {dimension}")
print(f"Index saved: {INDEX_FILE}")
print(f"Image list saved: {IMAGE_LIST_FILE}")