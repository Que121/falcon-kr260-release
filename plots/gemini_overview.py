#!/usr/bin/env python3
"""Generate the datapath overview via Gemini image generation (Nano Banana Pro = Gemini 3 Pro Image).
Needs GEMINI_API_KEY in the environment (get one at https://aistudio.google.com/apikey).
  python plots/gemini_overview.py [out.png]
Tries the Nano Banana Pro model first, falls back to other image-capable Gemini models.
After generating, VERIFY every label/number against fig2_datapath.svg (image models can mis-render text)."""
import os, sys

PROMPT = """A clean, flat, modern technical architecture diagram for an academic computer-vision paper.
White background, publication-quality, minimalist, sans-serif, thin arrows, soft shadows, no clutter.
A single horizontal left-to-right pipeline of 7 rounded-rectangle stages connected by arrows.
Title across the top: "WCET-certifiable camera-to-occupancy datapath on the Kria KR260".
The 7 stages, in order, each a labeled box with a colored header tag and small text underneath:
1. green box: "6 surround cameras"
2. blue box, tag "DPU INT8": "Image trunk + neck (ResNet-50 + CustomFPN)", subtext "45.5 ms/cam"
3. orange box, tag "PL IP #1 (ours)": "View transform (LSS gather)", subtext "WCET 4.7 ms, bit-exact"
4. blue box, tag "DPU INT8": "BEV encoder (CustomResNet + clamp 32)", subtext "98.3% mIoU"
5. orange box, tag "PL IP #2 (ours)": "BEV-neck upsample (bilinear resize x2)", subtext "WCET 3.44 ms, bit-exact"
6. grey box, tag "DPU + tiny CPU": "Occupancy head (final conv + softplus predicter)"
7. green box: "Occupancy 200x200x16"
A small legend row: blue = DPU (INT8 dense conv), orange = custom PL IP (our contribution),
grey = CPU tail, green = sensor / output.
A footer banner: "Both IPs placed & routed on the real K26: 211/230 MHz, 192/130 DSP - WCET-certifiable on programmable logic."
Use a calm professional palette (soft blue, soft orange, light grey, soft green). 16:6 aspect ratio.
All text must be crisp, correctly spelled, and exactly as written above."""

MODELS = ["gemini-3-pro-image-preview", "gemini-2.5-flash-image", "gemini-2.5-flash-image-preview",
          "gemini-2.0-flash-preview-image-generation", "gemini-2.0-flash-exp-image-generation"]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "figs/overview_gemini.png"
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("ERROR: set GEMINI_API_KEY (https://aistudio.google.com/apikey) and restart, then re-run.")
    from google import genai
    client = genai.Client(api_key=key)
    for m in MODELS:
        try:
            r = client.models.generate_content(model=m, contents=[PROMPT])
            for part in r.candidates[0].content.parts:
                data = getattr(getattr(part, "inline_data", None), "data", None)
                if data:
                    open(out, "wb").write(data)
                    print(f"OK model={m} -> {out} ({len(data)} bytes)")
                    return
            print(f"model {m}: no image in response (text only)")
        except Exception as e:
            print(f"model {m} failed: {str(e)[:120]}")
    sys.exit("No image-capable Gemini model worked with this key/region.")


if __name__ == "__main__":
    main()
