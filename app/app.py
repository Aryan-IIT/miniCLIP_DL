import sys
from pathlib import Path

"""Streamlit front-end for exploring MiniCLIP retrieval models."""

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, RobertaTokenizer

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
TRIAL_DIR = REPO_DIR / "trial"
sys.path.append(str(TRIAL_DIR))

from src.config import Config as TrainConfig
from src.model import VisionEncoder, TextEncoder, MiniCLIP


APP_DATA_DIR = APP_DIR / "data"
APP_MODELS_DIR = APP_DIR / "models"
APP_IMAGE_ROOT = APP_DATA_DIR / "flickr30k" / "Images"


# --- Helpers for locating models/embeddings ---
def list_available_checkpoints():
    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(APP_MODELS_DIR.glob("*.pt"))
    # Fallback to training checkpoints if app/models is empty
    if not checkpoints:
        checkpoints = [
            path for path in [
                TRIAL_DIR / "checkpoints_partial" / "best_miniclip_partial.pt",
                TRIAL_DIR / "checkpoints" / "best_miniclip.pt",
                TrainConfig.MODEL_SAVE_PATH,
            ] if path.exists()
        ]
    return checkpoints


def embeddings_path_for_checkpoint(ckpt_path: Path) -> Path:
    stem = ckpt_path.stem
    return APP_DATA_DIR / f"{stem}_embeddings.pt"


# Cache processor/tokenizer to avoid repeated HF downloads.
@st.cache_resource(show_spinner=False, hash_funcs={Path: str})
def load_processors():
    processor = ViTImageProcessor.from_pretrained(TrainConfig.VISION_MODEL_NAME)
    tokenizer = RobertaTokenizer.from_pretrained(TrainConfig.TEXT_MODEL)
    return processor, tokenizer


@st.cache_resource(show_spinner=True)
def load_model(checkpoint_path: str, model_type: str):
    ckpt_path = Path(checkpoint_path)
    vision_enc = VisionEncoder(model_type=model_type)
    text_enc = TextEncoder()
    model = MiniCLIP(vision_enc, text_enc)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        # Support older checkpoints that saved a plain `nn.Linear` as
        # `vision_encoder.projection` / `text_encoder.projection` by
        # remapping those keys into the new `ProjectionHead` submodules.
        mapped = False
        new_state = {}
        for k, v in state.items():
            if k.startswith("vision_encoder.projection.") and not k.startswith("vision_encoder.projection.projection"):
                # map `vision_encoder.projection.weight` -> `vision_encoder.projection.projection.weight`
                suffix = k.split("vision_encoder.projection.", 1)[1]
                new_key = f"vision_encoder.projection.projection.{suffix}"
                new_state[new_key] = v
                mapped = True
            elif k.startswith("text_encoder.projection.") and not k.startswith("text_encoder.projection.projection"):
                suffix = k.split("text_encoder.projection.", 1)[1]
                new_key = f"text_encoder.projection.projection.{suffix}"
                new_state[new_key] = v
                mapped = True
            else:
                new_state[k] = v

        # Load state dict non-strictly to allow missing new submodule params
        model.load_state_dict(new_state, strict=False)
        if mapped:
            st.sidebar.info("Loaded checkpoint (projection keys remapped to new ProjectionHead).")
        else:
            st.sidebar.success(f"Loaded weights from {ckpt_path.relative_to(REPO_DIR)}")
    else:
        st.sidebar.warning(f"Checkpoint {ckpt_path} not found; model uses random initialization.")

    model.eval()
    return model


@st.cache_resource(show_spinner=True)
def load_embeddings(embedding_path: str):
    embed_path = Path(embedding_path)
    if not embed_path.exists():
        return None
    data = torch.load(embed_path, map_location="cpu")
    payload = {}
    for key in ("image_embeddings", "text_embeddings"):
        payload[key] = torch.tensor(data[key]) if not torch.is_tensor(data[key]) else data[key]
        payload[key] = payload[key].float()
    payload["image_ids"] = data.get("image_ids", [])
    payload["captions"] = data.get("captions", [])
    return payload


def normalize_embeddings(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True)


def encode_image(model, processor, pil_image):
    inputs = processor(images=pil_image, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        img_emb = model.vision_encoder(inputs)
    return normalize_embeddings(img_emb)


def encode_text(model, tokenizer, text):
    tokens = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=TrainConfig.MAX_LEN,
        return_tensors="pt",
    )
    with torch.no_grad():
        txt_emb = model.text_encoder(tokens["input_ids"], tokens["attention_mask"])
    return normalize_embeddings(txt_emb)


def retrieve_top_k(query_emb, gallery_emb, k=10):
    sims = torch.matmul(query_emb, gallery_emb.T).squeeze(0)
    k = min(k, sims.shape[-1])
    values, indices = torch.topk(sims, k=k)
    return values.cpu().numpy(), indices.cpu().numpy()


def retrieve_top_k_unique(query_emb, gallery_emb, image_ids, k=5):
    """
    Retrieve top-k unique images (by image_id) for a query embedding.
    This avoids returning the same image multiple times when the gallery
    contains multiple captions per image.
    Returns (scores_list, image_id_list).
    """
    sims = torch.matmul(query_emb, gallery_emb.T).squeeze(0)
    # sort all scores descending
    sorted_vals, sorted_idx = sims.sort(descending=True)

    seen = set()
    selected_ids = []
    selected_scores = []
    for score, idx in zip(sorted_vals.tolist(), sorted_idx.tolist()):
        img_id = image_ids[idx]
        if img_id in seen:
            continue
        seen.add(img_id)
        selected_ids.append(img_id)
        selected_scores.append(score)
        if len(selected_ids) >= k:
            break

    return selected_scores, selected_ids


def display_caption_results(captions, indices, scores):
    rows = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        caption = captions[idx] if idx < len(captions) else f"[Unknown {idx}]"
        rows.append({"Rank": rank, "Caption": caption, "Score": float(score)})
    df = pd.DataFrame(rows)
    st.subheader("Top Captions")
    st.table(df)
    chart_df = df.set_index("Caption")[["Score"]]
    st.bar_chart(chart_df)


def display_image_grid(image_paths, scores=None, title="Top Images"):
    st.subheader(title)
    columns = st.columns(5)
    for idx, img_path in enumerate(image_paths):
        col = columns[idx % 5]
        with col:
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.text(img_path.name)
            if scores is not None:
                st.caption(f"Score: {scores[idx]:.3f}")
    if scores is not None:
        rows = [{"Image": path.name, "Score": float(score)} for path, score in zip(image_paths, scores)]
        chart_df = pd.DataFrame(rows).set_index("Image")
        st.bar_chart(chart_df)


def render_image_upload(model, processor, embeddings):
    # Simple upload flow that mirrors the CLI inference helper.
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Query Image", use_container_width=True)
        query_emb = encode_image(model, processor, image)
        values, indices = retrieve_top_k(query_emb, embeddings["text_embeddings"])
        display_caption_results(embeddings["captions"], indices, values)


def render_image_gallery(model, processor, embeddings):
    st.write("Select an example image to retrieve captions.")
    # Show distinct images in the gallery (dataset may contain multiple
    # captions per image, so `image_ids` can repeat). Keep order and pick
    # the first N unique images for the gallery.
    unique_image_ids = list(dict.fromkeys(embeddings["image_ids"]))
    sample_paths = [APP_IMAGE_ROOT / img for img in unique_image_ids[:10]]

    if "selected_gallery_index" not in st.session_state:
        st.session_state.selected_gallery_index = None

    cols = st.columns(5)
    selected_idx = None
    for idx, img_path in enumerate(sample_paths):
        col = cols[idx % 5]
        with col:
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.text(img_path.name)
            if st.button(f"Select {idx + 1}", key=f"gallery_{idx}"):
                st.session_state.selected_gallery_index = idx
                selected_idx = idx

    chosen_idx = st.session_state.selected_gallery_index if selected_idx is None else selected_idx
    if chosen_idx is not None and chosen_idx < len(sample_paths):
        img_name = embeddings["image_ids"][chosen_idx]
        st.markdown(f"**Selected Image:** {img_name}")
        image = Image.open(sample_paths[chosen_idx]).convert("RGB")
        st.image(image, use_container_width=True)
        query_emb = encode_image(model, processor, image)
        values, indices = retrieve_top_k(query_emb, embeddings["text_embeddings"])
        display_caption_results(embeddings["captions"], indices, values)


def render_text_query(model, tokenizer, embeddings):
    st.write("Enter a caption to retrieve the most relevant images.")
    caption_list = embeddings["captions"][:50]
    col1, col2 = st.columns(2)
    with col1:
        query_text = st.text_input("Custom caption")
    with col2:
        sample_choice = st.selectbox("Or pick a sample caption", ["None"] + caption_list)

    if sample_choice != "None":
        query_text = sample_choice
        st.info(f"Using sample caption: {query_text}")

    if query_text:
        # encode_text expects a list; it returns a (1, dim) tensor
        query_emb = encode_text(model, tokenizer, [query_text])
        # Retrieve top unique image ids (one image per id) and their scores
        scores, image_ids = retrieve_top_k_unique(
            query_emb,
            embeddings["image_embeddings"],
            embeddings["image_ids"],
            k=5,
        )
        top_image_paths = [APP_IMAGE_ROOT / img for img in image_ids]
        display_image_grid(top_image_paths, scores=scores)


def main():
    st.set_page_config(page_title="MiniCLIP Retrieval Demo", layout="wide")
    st.title("MiniCLIP Retrieval Explorer")
    st.sidebar.header("Configuration")

    # Let user choose vision backbone first (ViT or ResNet)
    backbone = st.sidebar.selectbox(
        "Vision Backbone",
        ["vit", "resnet"],
        format_func=lambda x: "ViT" if x == "vit" else "ResNet",
    )

    # Map backbone choice to preferred checkpoint filenames
    filename_map = {"vit": "vitsmall_roberta.pt", "resnet": "resnet101_roberta.pt"}
    preferred_name = filename_map.get(backbone)

    # Build candidate paths: prefer app/models, then trial checkpoints folders
    candidate_paths = []
    preferred_in_app = APP_MODELS_DIR / preferred_name
    if preferred_in_app.exists():
        candidate_paths.append(preferred_in_app)
    else:
        trial_candidates = [
            TRIAL_DIR / "checkpoints_partial" / preferred_name,
            TRIAL_DIR / "checkpoints" / preferred_name,
            TRIAL_DIR / "checkpoints_full" / preferred_name,
            TRIAL_DIR / "checkpoints_store" / preferred_name,
        ]
        for p in trial_candidates:
            if p.exists():
                candidate_paths.append(p)

    # Fallback: list all available checkpoints if specific one isn't found
    if not candidate_paths:
        candidate_paths = list_available_checkpoints()

    if not candidate_paths:
        st.error("No checkpoints found in app/models or trial checkpoints. Please add .pt files to app/models/.")
        return

    checkpoint_labels = [path.name for path in candidate_paths]
    selected_label = st.sidebar.selectbox("Select model checkpoint", checkpoint_labels)
    checkpoint_path = candidate_paths[checkpoint_labels.index(selected_label)]
    embedding_path = embeddings_path_for_checkpoint(checkpoint_path)

    embeddings = load_embeddings(str(embedding_path))
    if embeddings is None:
        st.error(
            (
                f"No embeddings found for {selected_label} at {embedding_path}.\n"
                f"Run `python app/prepare_embeddings.py --checkpoint {checkpoint_path} "
                f"--output {embedding_path}` to generate them."
            )
        )
        return

    if not APP_IMAGE_ROOT.exists():
        st.error(f"Image directory {APP_IMAGE_ROOT} is missing. Place the Flickr30k images there.")
        return

    model = load_model(str(checkpoint_path), backbone)
    processor, tokenizer = load_processors()

    mode = st.sidebar.radio("Mode", ["Image → Text Retrieval", "Text → Image Retrieval"])
    if mode == "Image → Text Retrieval":
        tab1, tab2 = st.tabs(["Upload Image", "Gallery Selection"])
        with tab1:
            render_image_upload(model, processor, embeddings)
        with tab2:
            render_image_gallery(model, processor, embeddings)
    else:
        render_text_query(model, tokenizer, embeddings)


if __name__ == "__main__":
    main()
