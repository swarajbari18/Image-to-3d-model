"""
Image-to-3D Pipeline
====================
Converts a marketing composite image (front + back side-by-side) into a
metric-scaled, manifold GLB suitable for browser rendering.

Pipeline stages:
  1. Gemini  — parse scene, locate front/back views, look up dimensions
  2. SAM2    — segment front and back panels with Gemini feedback loop
  3. pix2gestalt — amodal-complete the occluded back panel
  4. Hunyuan3D-2mv — generate textured 3-D shape from front + back RGBA
  5. mesh_processing — clean, orient, scale to mm, export GLB
"""
