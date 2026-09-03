import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import os


MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"


class ImageCaptioningApp:

    def __init__(self, root):
        self.root = root

        self.root.title("CODSOFT - Image Captioning AI")
        self.root.geometry("850x750")
        self.root.minsize(750, 650)

        self.processor = None
        self.tokenizer = None
        self.model = None

        self.image_path = None
        self.photo = None

        # ---------------- HEADER ----------------

        header = tk.Frame(
            root,
            bg="#1f2937",
            height=110
        )
        header.pack(fill=tk.X)

        title = tk.Label(
            header,
            text="🖼️ IMAGE CAPTIONING AI",
            font=("Arial", 25, "bold"),
            bg="#1f2937",
            fg="white"
        )
        title.pack(pady=(20, 5))

        subtitle = tk.Label(
            header,
            text="Vision Transformer + GPT-2",
            font=("Arial", 12),
            bg="#1f2937",
            fg="white"
        )
        subtitle.pack()

        # ---------------- STATUS ----------------

        self.status = tk.Label(
            root,
            text="Loading AI model...",
            font=("Arial", 12, "bold")
        )
        self.status.pack(pady=12)

        # ---------------- IMAGE AREA ----------------

        image_frame = tk.Frame(
            root,
            bd=2,
            relief=tk.GROOVE,
            width=650,
            height=400
        )
        image_frame.pack(
            padx=30,
            pady=10,
            fill=tk.BOTH,
            expand=True
        )

        image_frame.pack_propagate(False)

        self.image_label = tk.Label(
            image_frame,
            text="No image selected\n\nClick 'Select Image' to choose an image",
            font=("Arial", 15)
        )
        self.image_label.pack(
            expand=True
        )

        # ---------------- SELECT BUTTON ----------------

        select_button = tk.Button(
            root,
            text="📁 SELECT IMAGE",
            font=("Arial", 13, "bold"),
            command=self.select_image,
            padx=25,
            pady=10
        )
        select_button.pack(pady=10)

        # ---------------- CAPTION BUTTON ----------------

        self.caption_button = tk.Button(
            root,
            text="✨ GENERATE CAPTION",
            font=("Arial", 13, "bold"),
            command=self.generate_caption,
            padx=25,
            pady=10,
            state=tk.DISABLED
        )
        self.caption_button.pack(pady=5)

        # ---------------- CAPTION AREA ----------------

        caption_title = tk.Label(
            root,
            text="Generated Caption",
            font=("Arial", 15, "bold")
        )
        caption_title.pack(pady=(15, 5))

        self.caption_label = tk.Label(
            root,
            text="Your AI-generated caption will appear here.",
            font=("Arial", 13),
            wraplength=750,
            justify=tk.CENTER
        )
        self.caption_label.pack(
            padx=20,
            pady=5
        )

        # ---------------- CLEAR BUTTON ----------------

        clear_button = tk.Button(
            root,
            text="🔄 CLEAR",
            font=("Arial", 11),
            command=self.clear,
            padx=25,
            pady=6
        )
        clear_button.pack(pady=10)

        # Load model after GUI appears
        self.root.after(100, self.load_model)

    # ---------------- LOAD MODEL ----------------

    def load_model(self):

        try:
            self.status.config(
                text="Loading AI model... Please wait."
            )

            self.root.update()

            self.processor = ViTImageProcessor.from_pretrained(
                MODEL_NAME
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME
            )

            self.model = VisionEncoderDecoderModel.from_pretrained(
                MODEL_NAME
            )

            self.status.config(
                text="✅ AI model loaded. Select an image."
            )

        except Exception as error:

            self.status.config(
                text="❌ Failed to load AI model."
            )

            messagebox.showerror(
                "Model Error",
                f"Could not load the AI model.\n\n{error}"
            )

    # ---------------- SELECT IMAGE ----------------

    def select_image(self):

        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        if not os.path.isfile(file_path):
            return

        try:

            image = Image.open(file_path).convert("RGB")

            self.image_path = file_path

            # Resize image for preview
            image.thumbnail((600, 350))

            self.photo = ImageTk.PhotoImage(image)

            self.image_label.config(
                image=self.photo,
                text=""
            )

            self.caption_button.config(
                state=tk.NORMAL
            )

            self.caption_label.config(
                text="Click 'Generate Caption' to analyze the image."
            )

            self.status.config(
                text=f"Selected: {os.path.basename(file_path)}"
            )

        except Exception as error:

            messagebox.showerror(
                "Image Error",
                f"Could not open the image.\n\n{error}"
            )

    # ---------------- GENERATE CAPTION ----------------

    def generate_caption(self):

        if not self.image_path:
            messagebox.showwarning(
                "No Image",
                "Please select an image first."
            )
            return

        if self.model is None:
            messagebox.showwarning(
                "Model Not Ready",
                "Please wait for the AI model to finish loading."
            )
            return

        try:

            self.status.config(
                text="🔍 Analyzing image..."
            )

            self.caption_button.config(
                state=tk.DISABLED
            )

            self.root.update()

            image = Image.open(
                self.image_path
            ).convert("RGB")

            pixel_values = self.processor(
                images=image,
                return_tensors="pt"
            ).pixel_values

            with torch.no_grad():

                output_ids = self.model.generate(
                    pixel_values,
                    max_length=50,
                    num_beams=4
                )

            caption = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            self.caption_label.config(
                text=caption
            )

            self.status.config(
                text="✅ Caption generated successfully!"
            )

        except Exception as error:

            self.status.config(
                text="❌ Could not process the image."
            )

            messagebox.showerror(
                "Caption Error",
                str(error)
            )

        finally:

            self.caption_button.config(
                state=tk.NORMAL
            )

    # ---------------- CLEAR ----------------

    def clear(self):

        self.image_path = None
        self.photo = None

        self.image_label.config(
            image="",
            text="No image selected\n\nClick 'Select Image' to choose an image"
        )

        self.caption_label.config(
            text="Your AI-generated caption will appear here."
        )

        self.caption_button.config(
            state=tk.DISABLED
        )

        self.status.config(
            text="Ready. Select an image."


        )


# ---------------- START PROGRAM ----------------

if __name__ == "__main__":

    root = tk.Tk()

    app = ImageCaptioningApp(root)

    root.mainloop()