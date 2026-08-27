from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import os


MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"


def load_model():
    print("Loading image captioning AI model...")
    print("This may take a moment on the first run.\n")

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    return processor, tokenizer, model


def generate_caption(image_path, processor, tokenizer, model):
    image = Image.open(image_path).convert("RGB")

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=50,
            num_beams=4
        )

    caption = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return caption


def main():
    print("=" * 60)
    print("       CODSOFT - IMAGE CAPTIONING AI")
    print("=" * 60)

    print("\nThis system uses a pre-trained Vision Transformer")
    print("and GPT-2 language model to generate image captions.\n")

    processor, tokenizer, model = load_model()

    while True:
        print("\n" + "-" * 60)

        image_path = input(
            "Enter image path (or type 'exit' to quit): "
        ).strip()

        if image_path.lower() in ["exit", "quit", "bye"]:
            print("\nThank you for using the Image Captioning AI! 👋")
            break

        image_path = image_path.strip('"').strip("'")

        if not os.path.isfile(image_path):
            print("\n❌ Image file not found.")
            continue

        try:
            print("\n🔍 Analyzing image...")

            caption = generate_caption(
                image_path,
                processor,
                tokenizer,
                model
            )

            print("\n🖼️ Generated Caption:")
            print("-" * 60)
            print(caption)
            print("-" * 60)

        except Exception as error:
            print("\n❌ Could not process the image.")
            print("Error:", error)


if __name__ == "__main__":
    main()