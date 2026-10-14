import pickle

import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# -----------------------------
# Files
# -----------------------------

INDEX_FILE = "screenmind.index"
IMAGE_LIST_FILE = "images.pkl"

MODEL_NAME = "openai/clip-vit-base-patch32"


# -----------------------------
# Load CLIP
# -----------------------------

print("Loading ScreenMind...")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

print("CLIP loaded.")


# -----------------------------
# Load FAISS index
# -----------------------------

index = faiss.read_index(INDEX_FILE)

with open(IMAGE_LIST_FILE, "rb") as f:
    image_paths = pickle.load(f)

print(f"Loaded {len(image_paths)} screenshots.\n")


# -----------------------------
# Search loop
# -----------------------------

while True:

    query = input(
        "What are you looking for? "
        "(type 'exit' to quit): "
    )

    if query.lower() == "exit":
        break

    # Convert query to CLIP text input
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    )

    # Generate text embedding
    with torch.no_grad():

        text_embedding = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        text_embedding = text_embedding.pooler_output

    # Normalize
    text_embedding = text_embedding / text_embedding.norm(
        dim=-1,
        keepdim=True
    )

    # Convert to NumPy
    query_vector = text_embedding.cpu().numpy()

    # Search FAISS
    scores, indices = index.search(
        query_vector,
        min(5, len(image_paths))
    )

    # Display results
    print("\n==============================")
    print(" SCREENMIND SEARCH RESULTS")
    print("==============================")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        print(
            f"{rank}. "
            f"{image_paths[idx]} "
            f"(score: {score:.4f})"
        )

    print()