import os
import webbrowser
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_matplotlib_png(output_png="clean_pipeline_guide.png"):
    fig = plt.figure(figsize=(28, 16), dpi=220)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # Header
    ax.text(14, 15.3, "CatVTON Virtual Try-On: 0% to 100% Architecture & Data Flow", 
            fontsize=24, fontweight="bold", ha="center", color="#0F172A")
    ax.text(14, 14.7, "Concrete Data Shapes, Pixel Transformations, Memory Optimizations, and Attention Mechanics", 
            fontsize=13, ha="center", color="#475569")

    # Card Drawing Helper
    def draw_card(x, y, w, h, step_num, title, tool, content, example_box, border_c, bg_c):
        # Outer Card
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2,rounding_size=0.25",
            facecolor=bg_c, edgecolor=border_c, linewidth=2.5
        )
        ax.add_patch(box)

        # Step Badge
        badge = patches.FancyBboxPatch(
            (x + 0.3, y + h - 0.8), 2.0, 0.55,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=border_c, edgecolor="none"
        )
        ax.add_patch(badge)
        ax.text(x + 1.3, y + h - 0.52, f"STEP {step_num}", color="#FFFFFF", fontsize=11, fontweight="bold", ha="center", va="center")

        # Title & Tool
        ax.text(x + 2.5, y + h - 0.52, title, fontsize=13, fontweight="bold", color="#0F172A", va="center")
        ax.text(x + w - 0.4, y + h - 0.52, f"[{tool}]", fontsize=11, fontweight="bold", color=border_c, ha="right", va="center")

        # Main Explanation Content
        ax.text(x + 0.4, y + h - 1.2, content, fontsize=11, color="#1E293B", va="top", linespacing=1.4)

        # Concrete Example Box
        ex_bg = patches.FancyBboxPatch(
            (x + 0.3, y + 0.3), w - 0.6, 1.8,
            boxstyle="round,pad=0.15,rounding_size=0.15",
            facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5
        )
        ax.add_patch(ex_bg)
        ax.text(x + 0.5, y + 1.8, "CONCRETE EXAMPLE / DATA:", fontsize=9.5, fontweight="bold", color="#64748B")
        ax.text(x + 0.5, y + 1.0, example_box, fontsize=10.5, color="#0F172A", va="center", linespacing=1.35)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#334155", lw=3, mutation_scale=20)
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.25, label, fontsize=10.5, fontweight="bold", color="#0F172A", ha="center")

    # ================= ROW 1 (TOP) =================
    draw_card(
        0.8, 7.8, 6.0, 6.2, "1", "Pose & Background Removal", "rembg + mediapipe",
        "* User uploads Person (384x512) and Garment.\n"
        "* rembg strips backdrop around standalone garment.\n"
        "* mediapipe calculates 33 skeleton body joints\n"
        "  to capture body posture, rotation, and arm tilt.",
        "Joint Coordinates:\n"
        "-> L_Shoulder: (X=140, Y=150, Z=-0.12)\n"
        "-> R_Shoulder: (X=244, Y=150, Z=+0.05)\n"
        "-> Body Orientation: Facing Forward",
        "#2563EB", "#EFF6FF"
    )

    draw_card(
        7.6, 7.8, 6.0, 6.2, "2", "SCHP Clothing Segmentation", "SCHP + OpenCV",
        "* SCHP neural network classifies all 196,608 pixels.\n"
        "* Labels: 13=Face, 5=Upper Shirt, 9=Pants, 2=Hair.\n"
        "* Label 5 becomes binary mask (255=White, 0=Black).\n"
        "* cv2.dilate adds 15px border to eliminate old collar.",
        "Pixel Tagging Matrix:\n"
        "[[ 13 (Face),  13 (Face)  ] -> Mask = 0 (Keep)\n"
        " [  5 (Shirt),  5 (Shirt) ] -> Mask = 255 (Erase)]\n"
        "Result: Agnostic mask isolating chest area.",
        "#059669", "#ECFDF5"
    )

    draw_card(
        14.4, 7.8, 6.0, 6.2, "3", "Latent VAE Compression", "SD-VAE Encoder",
        "* Full RGB image (384x512x3) has 196,608 values.\n"
        "* Processing raw pixels crashes GPU memory.\n"
        "* VAE Encoder compresses 8x into latent representations.\n"
        "* Person Latent: (4, 64, 48) | Garment: (4, 64, 48).",
        "Compression Math:\n"
        "8x8 RGB block [240, 120, 80] becomes 4 floats:\n"
        "[-0.482, +1.104, -0.193, +0.841]\n"
        "Cuts RAM usage by 98% (3,072 values).",
        "#7C3AED", "#F5F3FF"
    )

    draw_card(
        21.2, 7.8, 6.0, 6.2, "4", "Vertical Spatial Stacking", "model/pipeline.py",
        "* Stacks both latent representations vertically:\n"
        "  - Top Half: Person Latent + Gaussian Static Noise\n"
        "  - Bottom Half: Clean Garment Reference Latent\n"
        "* Glued into one unified canvas of shape (4, 128, 48).\n"
        "* Unlocks cross-image feature exchange in UNet.",
        "Unified Canvas Matrix (4, 128, 48):\n"
        "[ Rows   0-63 ]: Masked Person + Noise (To Paint)\n"
        "[ Rows 64-127 ]: Reference Garment (To Copy)\n"
        "Total Latent Tokens = 6,144 spatial cells.",
        "#D97706", "#FFFBEB"
    )

    # Row 1 Arrows
    draw_arrow(6.8, 10.9, 7.6, 10.9)
    draw_arrow(13.6, 10.9, 14.4, 10.9)
    draw_arrow(20.4, 10.9, 21.2, 10.9)
    draw_arrow(24.2, 7.8, 24.2, 7.0, "Input to UNet")

    # ================= ROW 2 (BOTTOM) =================
    draw_card(
        21.2, 0.6, 6.0, 6.2, "5", "Attention Feature Transfer", "model/attn_processor.py",
        "* UNet runs 20-step DDIM denoising on noise canvas.\n"
        "* Self-Attention connects top query to bottom key.\n"
        "* Top Half Query: 'What texture belongs on chest?'\n"
        "* Bottom Half Key: 'Matches red plaid fabric at 0.94'.\n"
        "* Chunk size (2048) prevents the 36 GiB memory overflow.",
        "Attention Matrix Score:\n"
        "Score = Softmax( Q * K.T / sqrt(dim) )\n"
        "Match Score = 0.94 -> Injects fabric texture.\n"
        "Peak Attention Memory: < 500 MB (Stable).",
        "#DC2626", "#FEF2F2"
    )

    draw_card(
        14.4, 0.6, 6.0, 6.2, "6", "Slicing & VAE Decoding", "SD-VAE Decoder",
        "* After 20 denoising steps, noise is fully removed.\n"
        "* Slices canvas across horizontal midline:\n"
        "  - Top 64 rows: Kept (Finished try-on latent)\n"
        "  - Bottom 64 rows: Discarded (Reference shirt)\n"
        "* VAE Decoder unzips (4, 64, 48) back to full RGB.",
        "Reconstruction:\n"
        "Latent (4, 64, 48) -> Multiplied by (1 / 0.18215)\n"
        "-> VAE Decoder -> Clamped [0, 1] RGB\n"
        "Output Shape: (384, 512, 3) Color Image.",
        "#DB2777", "#FDF2F8"
    )

    draw_card(
        7.6, 0.6, 6.0, 6.2, "7", "Face & Identity Preservation", "postprocess.py",
        "* 512px diffusion can soften fine facial details.\n"
        "* Extracts original face from Step 1 (Labels 1,2,13).\n"
        "* Smooths boundary using 15x15 Gaussian blur.\n"
        "* Composites real face cleanly over AI clothing.",
        "Alpha Blending Formula:\n"
        "Final_Pixel = (Original_Face * Alpha) +\n"
        "              (TryOn_Result * (1.0 - Alpha))\n"
        "Result: 100% facial sharpness retained.",
        "#0D9488", "#F0FDFA"
    )

    draw_card(
        0.8, 0.6, 6.0, 6.2, "8", "Final Verified Output", "output.png / Gradio",
        "* Output image is saved to disk or shown in Web App.\n"
        "* Clothing matches realistic folds, shadows, and body.\n"
        "* Total Execution Time: ~25 seconds on Apple Silicon.\n"
        "* Peak System Memory: < 4 GB (Zero system freeze).",
        "Pipeline Checklist:\n"
        "[+] Garment Warping: Passed\n"
        "[+] Memory Footprint: Passed (FP16 MPS)\n"
        "[+] Identity Retention: Passed\n"
        "[+] File: output.png",
        "#16A34A", "#F0FDF4"
    )

    # Row 2 Arrows
    draw_arrow(21.2, 3.7, 20.4, 3.7)
    draw_arrow(14.4, 3.7, 13.6, 3.7)
    draw_arrow(7.6, 3.7, 6.8, 3.7)

    plt.tight_layout()
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    print(f"Clean PNG diagram saved to: {output_png}")


def generate_html_diagram(output_html="pipeline_guide.html"):
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CatVTON Virtual Try-On Pipeline Visualizer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0B0F19; color: #E2E8F0; font-family: system-ui, -apple-system, sans-serif; }
        .card { background: #151C2E; border: 1px solid #23304D; border-radius: 12px; transition: transform 0.2s; }
        .card:hover { transform: translateY(-3px); border-color: #3B82F6; }
        .code-box { background: #0A0D14; border: 1px solid #1E293B; border-radius: 8px; font-family: ui-monospace, monospace; }
    </style>
</head>
<body class="p-8">
    <div class="max-w-7xl mx-auto">
        <div class="text-center mb-10">
            <h1 class="text-3xl font-bold text-white mb-2">CatVTON Virtual Try-On Architecture (0% to 100%)</h1>
            <p class="text-slate-400 text-sm">Interactive Visual Guide of Pixel Transformations, Latent Stacking, and Attention Mechanics</p>
        </div>

        <!-- Flow Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <!-- Step 1 -->
            <div class="card p-5">
                <div class="flex justify-between items-center mb-3">
                    <span class="bg-blue-600 text-white text-xs font-bold px-2.5 py-1 rounded">STEP 1</span>
                    <span class="text-xs text-blue-400 font-mono font-bold">rembg + mediapipe</span>
                </div>
                <h3 class="font-bold text-lg text-white mb-2">Pose & Cutout</h3>
                <p class="text-xs text-slate-300 mb-4 leading-relaxed">
                    Strips background around garment and detects 33 skeleton keypoints to measure shoulder tilt and arm angles.
                </p>
                <div class="code-box p-3 text-xs text-blue-300">
                    <strong>Skeleton Keypoints:</strong><br>
                    Shoulder_L: (140, 150)<br>
                    Shoulder_R: (244, 150)<br>
                    Status: Facing Forward
                </div>
            </div>

            <!-- Step 2 -->
            <div class="card p-5">
                <div class="flex justify-between items-center mb-3">
                    <span class="bg-emerald-600 text-white text-xs font-bold px-2.5 py-1 rounded">STEP 2</span>
                    <span class="text-xs text-emerald-400 font-mono font-bold">SCHP + OpenCV</span>
                </div>
                <h3 class="font-bold text-lg text-white mb-2">Erase Old Shirt</h3>
                <p class="text-xs text-slate-300 mb-4 leading-relaxed">
                    SCHP tags pixel IDs (5=Shirt, 13=Face). Label 5 is extracted and dilated by 15px to build the inpainting mask.
                </p>
                <div class="code-box p-3 text-xs text-emerald-300">
                    <strong>Agnostic Mask:</strong><br>
                    Face/Pants: Set to 0 (Keep)<br>
                    Shirt Area: Set to 255 (Erase)<br>
                    Dilate: +15px safety border
                </div>
            </div>

            <!-- Step 3 -->
            <div class="card p-5">
                <div class="flex justify-between items-center mb-3">
                    <span class="bg-purple-600 text-white text-xs font-bold px-2.5 py-1 rounded">STEP 3</span>
                    <span class="text-xs text-purple-400 font-mono font-bold">SD-VAE Encoder</span>
                </div>
                <h3 class="font-bold text-lg text-white mb-2">VAE Compression</h3>
                <p class="text-xs text-slate-300 mb-4 leading-relaxed">
                    Compresses 196,608 raw RGB pixels 8x into compact latent tensors of shape (4, 64, 48) to save 98% memory.
                </p>
                <div class="code-box p-3 text-xs text-purple-300">
                    <strong>Tensor Shapes:</strong><br>
                    Raw: (3, 384, 512) RGB<br>
                    Latent: (4, 64, 48)<br>
                    Values: [-0.48, +1.10, -0.19]
                </div>
            </div>

            <!-- Step 4 -->
            <div class="card p-5">
                <div class="flex justify-between items-center mb-3">
                    <span class="bg-amber-600 text-white text-xs font-bold px-2.5 py-1 rounded">STEP 4</span>
                    <span class="text-xs text-amber-400 font-mono font-bold">model/pipeline.py</span>
                </div>
                <h3 class="font-bold text-lg text-white mb-2">Canvas Stacking</h3>
                <p class="text-xs text-slate-300 mb-4 leading-relaxed">
                    Glues the masked person latent (with random noise) and garment reference vertically into one unified matrix.
                </p>
                <div class="code-box p-3 text-xs text-amber-300">
                    <strong>Canvas (4, 128, 48):</strong><br>
                    Top (0-63): Person + Noise<br>
                    Bottom (64-127): Reference Shirt<br>
                    Total Tokens: 6,144
                </div>
            </div>

            <!-- Step 5 -->
            <div class="card p-5">
                <div class="flex justify-between items-center mb-3">
                    <span class="bg-rose-600 text-white text-xs font-bold px-2.5 py-1 rounded">STEP 5</span>
                    <span class="text-xs text-rose-400 font-mono font-bold">Attention Processor</span>
                </div>
                <h3 class="font-bold text-lg text-white mb-2">Feature Transfer</h3>
                <p class="text-xs text-slate-300 mb-4 leading-relaxed">
                    UNet denoising loop where body Query searches garment Key to pull fabric texture. 2048 chunking prevents memory crashes.
                </p>
                <div class="code-box p-3 text-xs text-rose-300">
                    <strong>Attention Math:</strong><br>
                    Score = Softmax(Q * K^T / sqrt(d))<br>
                    Match: 0.94 -> Injects Plaid<br>
                    Peak Memory: &lt; 500 MB
                </div>
            </div>

            <!-- Step 6 -->
            <div class="card p-5">
                <div class="flex justify-between items-center mb-3">
                    <span class="bg-pink-600 text-white text-xs font-bold px-2.5 py-1 rounded">STEP 6</span>
                    <span class="text-xs text-pink-400 font-mono font-bold">SD-VAE Decoder</span>
                </div>
                <h3 class="font-bold text-lg text-white mb-2">Unzip to Pixels</h3>
                <p class="text-xs text-slate-300 mb-4 leading-relaxed">
                    Discards bottom reference half, takes top try-on latent (4, 64, 48), and decodes it back to full 384x512 RGB.
                </p>
                <div class="code-box p-3 text-xs text-pink-300">
                    <strong>Decoding Step:</strong><br>
                    Split: Top 64 rows kept<br>
                    Decode: Latent * (1/0.18215)<br>
                    Output: (3, 384, 512) Image
                </div>
            </div>

            <!-- Step 7 -->
            <div class="card p-5">
                <div class="flex justify-between items-center mb-3">
                    <span class="bg-teal-600 text-white text-xs font-bold px-2.5 py-1 rounded">STEP 7</span>
                    <span class="text-xs text-teal-400 font-mono font-bold">postprocess.py</span>
                </div>
                <h3 class="font-bold text-lg text-white mb-2">Face Lock</h3>
                <p class="text-xs text-slate-300 mb-4 leading-relaxed">
                    Uses OpenCV Gaussian alpha mask to blend original crisp face and hair over the newly generated clothing.
                </p>
                <div class="code-box p-3 text-xs text-teal-300">
                    <strong>Alpha Compositing:</strong><br>
                    Result = (Face * Alpha) + <br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Tryon * (1 - Alpha))<br>
                    Face Blur: 0% (Crisp)
                </div>
            </div>

            <!-- Step 8 -->
            <div class="card p-5">
                <div class="flex justify-between items-center mb-3">
                    <span class="bg-green-600 text-white text-xs font-bold px-2.5 py-1 rounded">STEP 8</span>
                    <span class="text-xs text-green-400 font-mono font-bold">output.png</span>
                </div>
                <h3 class="font-bold text-lg text-white mb-2">Final Output</h3>
                <p class="text-xs text-slate-300 mb-4 leading-relaxed">
                    Fitting completes with realistic folds and shadows matching body posture. Ready for Web App UI integration.
                </p>
                <div class="code-box p-3 text-xs text-green-300">
                    <strong>Audit Status:</strong><br>
                    Time: ~25s (FP16 MPS)<br>
                    RAM: &lt; 4 GB Peak<br>
                    File: Saved to output.png
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
    with open(output_html, "w") as f:
        f.write(html_content)
    print(f"Interactive HTML visualizer saved to: {output_html}")


if __name__ == "__main__":
    generate_matplotlib_png()
    generate_html_diagram()
    # Auto-open HTML in browser for maximum clarity
    webbrowser.open("file://" + os.path.abspath("pipeline_guide.html"))
