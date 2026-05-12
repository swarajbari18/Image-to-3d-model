# Mesh Processing & Metric Scaling Research
**Context:** Post-processing Hunyuan3D-2mv GLB output (~50-300K triangles) on CPU-only Colab.
**Goal:** Clean → orient canonically → scale to mm → export clean GLB.
**Research Date:** May 2026

---

## 1. Library Installation

### Recommended install block for Colab

```bash
pip install "trimesh[easy]" pymeshfix pymeshlab open3d xatlas
```

**What each installs:**

| Package | Latest (May 2026) | Notes |
|---|---|---|
| `trimesh[easy]` | 4.12.x | `[easy]` adds pillow, scipy, networkx, shapely, rtree, lxml — all needed for textures + GLB |
| `pymeshfix` | 0.18.1 (Jan 2026) | Wraps MeshFix v2.1; requires numpy, optionally pyvista |
| `pymeshlab` | 2025.7.post1 (Jan 2026) | Python ≥ 3.7 64-bit; no compilation needed, pure wheel |
| `open3d` | 0.19.x | Large package (~500 MB); only needed for OBB via PCA |
| `xatlas` | 0.0.11 (Oct 2025) | Python bindings for C++ xatlas UV unwrapper |

### trimesh[easy] vs plain trimesh

- **Plain `pip install trimesh`:** Loads GLB/GLTF fine (only hard dep is numpy). But texture handling requires Pillow, which is NOT included.
- **`pip install trimesh[easy]`:** Adds Pillow (textures), scipy (convex hull, OBB), networkx (graph ops), rtree, shapely, lxml — use this for GLB with textures.
- GLB loading itself works without extras; texture decoding requires Pillow.

### PyMeshLab on Colab — known issues

- Installs fine on Colab (Linux x86_64, 64-bit Python 3.10+).
- **No known conflicts** with trimesh or pymeshfix.
- Fails only on 32-bit Python — not an issue on Colab.
- **Critical:** PyMeshLab **cannot save GLB/GLTF**. It can load GLB but export only to PLY/OBJ/STL/etc. Use trimesh for final GLB output.
- Version naming: `pymeshlab-2025.7` corresponds to PyMeshLab 2025.7 release.

### open3d on Colab

- Large install. Only needed here for `get_oriented_bounding_box()` (PCA-based OBB).
- Can be skipped if you implement PCA orientation via numpy/scipy directly (shown in Section 4).

---

## 2. Loading GLB with Textures in trimesh

### trimesh.load() vs trimesh.load_mesh()

```python
import trimesh

# CORRECT for GLB — returns a trimesh.Scene (multiple geometries + transforms)
scene = trimesh.load("output.glb")

# trimesh.load_mesh() forces single-mesh return — LOSES textures and scene graph
# Do NOT use force='mesh' if you want to keep PBR textures
mesh = trimesh.load_mesh("output.glb")  # strips scene, may lose materials
```

**Rule:** Always use `trimesh.load()` for GLB. It returns a `trimesh.Scene` object.

### Accessing mesh + texture from a loaded scene

```python
scene = trimesh.load("hunyuan_output.glb")

# scene.geometry is a dict: {name: trimesh.Trimesh, ...}
print(scene.geometry.keys())

# Get all meshes
meshes = list(scene.geometry.values())

# For single-body GLBs from Hunyuan3D, there's typically one or a few meshes
mesh = meshes[0]

# Access material/visual
print(type(mesh.visual))
# -> trimesh.visual.TextureVisuals  (if UV-mapped texture)
# -> trimesh.visual.ColorVisuals    (if vertex colors)

# Access UV coords
if hasattr(mesh.visual, 'uv'):
    uvs = mesh.visual.uv           # (N, 2) float array
    material = mesh.visual.material  # PBRMaterial or SimpleMaterial

# Access base color texture image
if hasattr(mesh.visual.material, 'baseColorTexture'):
    img = mesh.visual.material.baseColorTexture  # PIL Image
```

### Does trimesh preserve PBR textures from Hunyuan3D GLBs?

**Mostly yes, with caveats:**

- Hunyuan3D-2mv outputs a GLB with UV-mapped textures (not vertex colors). The texture is baked via Hunyuan3D-Paint as a high-res image embedded in the GLB.
- `trimesh.load()` will load this as `TextureVisuals` with a `PBRMaterial`.
- **Known issue (GitHub #2304):** Loading Objaverse-style GLBs with `force='mesh'` drops textures. Never use `force='mesh'` for textured GLBs.
- **Known issue:** Some GLBs with complex material graphs (e.g., metallic/roughness packed in separate channels) may not round-trip perfectly through trimesh. Basecolor texture is reliably preserved.
- `process=False` is recommended when loading to avoid auto-cleanup that can break UV seams:

```python
scene = trimesh.load("hunyuan_output.glb", process=False)
```

### Converting Scene to single Trimesh (for processing)

```python
# Modern API (trimesh 4.x) — to_mesh() bakes scene graph transforms
mesh = scene.to_mesh()
# WARNING: this MERGES all sub-meshes; UVs may be concatenated

# Alternative: work on the largest geometry directly
largest = max(scene.geometry.values(), key=lambda m: m.area)
```

---

## 3. Full Cleanup Pipeline

### 3a. Remove disconnected small components (floaters)

**Using trimesh.graph.split():**

```python
import trimesh
import numpy as np

def remove_small_components(mesh, area_threshold=0.01):
    """
    Remove disconnected components whose surface area is less than
    area_threshold * total_area of the largest component.
    Returns cleaned mesh.
    """
    components = mesh.split(only_watertight=False)
    if len(components) == 1:
        return mesh

    # Sort by area descending
    components = sorted(components, key=lambda c: c.area, reverse=True)
    largest_area = components[0].area

    # Keep only components above threshold
    kept = [c for c in components if c.area >= area_threshold * largest_area]

    if len(kept) == 1:
        return kept[0]

    # Concatenate kept components
    return trimesh.util.concatenate(kept)

mesh = remove_small_components(mesh, area_threshold=0.01)  # < 1% of largest = floater
```

**Notes:**
- `mesh.split()` returns list of Trimesh objects split by face connectivity.
- `only_watertight=False` keeps all components (including non-watertight ones) so we can filter manually.
- `trimesh.util.concatenate()` merges multiple Trimesh objects into one.
- Face-count threshold is an alternative: `[c for c in components if c.faces.shape[0] > min_faces]`

**Using PyMeshLab (more robust, handles large meshes faster):**

```python
import pymeshlab as pml

ms = pml.MeshSet()
ms.load_new_mesh("input.glb", load_in_a_single_layer=True)  # merge all layers

# Remove components smaller than X% of bounding box diagonal
ms.meshing_remove_connected_component_by_diameter(
    mincomponentdiag=pml.PercentageValue(5)  # remove if < 5% of bbox diagonal
)

# OR remove components with fewer than N faces
ms.meshing_remove_connected_component_by_face_number(mincomponentsize=100)
```

### 3b. pymeshfix — fix non-manifold + make watertight

**Exact API (no pyvista required — use PyTMesh):**

```python
import pymeshfix
import numpy as np

def fix_mesh_with_pymeshfix(mesh):
    """
    Fix non-manifold edges, self-intersections, and make mesh watertight.
    Returns a new trimesh.Trimesh.
    """
    tin = pymeshfix.PyTMesh()
    tin.load_array(mesh.vertices, mesh.faces)

    # Fix connectivity issues (non-manifold edges/vertices)
    tin.fix_connectivity()

    # Remove self-intersections (can be slow on dense meshes)
    # tin.strong_intersection_removal()  # optional, slow

    # Fill small boundary holes
    tin.fill_small_boundaries(refine=True)

    # Extract repaired mesh
    verts, faces = tin.return_arrays()
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

# Usage
fixed_mesh = fix_mesh_with_pymeshfix(mesh)
```

**Alternative — MeshFix class (requires pyvista):**

```python
import pymeshfix

# If pyvista is available
meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
meshfix.repair(verbose=True)
fixed_mesh = trimesh.Trimesh(
    vertices=meshfix.points,
    faces=meshfix.faces.reshape(-1, 3),
    process=False
)
```

**What pymeshfix actually repairs:**
- Non-manifold edges (T-junctions, bowtie vertices)
- Self-intersections
- Degenerate triangles (zero-area, collinear)
- Small holes via fan-fill

**What it does NOT do:** Isotropic remeshing, decimation, smoothing — use PyMeshLab for those.

### 3c. Taubin smoothing

**trimesh does NOT have Taubin smoothing.** `trimesh.smoothing` has Laplacian only:

```python
# trimesh Laplacian smooth (NOT Taubin — will shrink the mesh)
trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=10)
```

**Use PyMeshLab for Taubin (recommended — no shrinkage):**

```python
ms.apply_coord_taubin_smoothing(
    lambda_=0.5,    # positive, controls smoothing magnitude
    mu=-0.53,       # negative, controls anti-shrinking step
    stepsmoothnum=10,  # number of iterations
    selected=False   # apply to all faces
)
```

**Or use Open3D:**

```python
import open3d as o3d

o3d_mesh = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(mesh.vertices),
    triangles=o3d.utility.Vector3iVector(mesh.faces)
)
smoothed = o3d_mesh.filter_smooth_taubin(number_of_iterations=10, lambda_filter=0.5, mu=-0.53)
mesh.vertices = np.asarray(smoothed.vertices)
```

### 3d. Isotropic remeshing via PyMeshLab

**Exact filter name:** `meshing_isotropic_explicit_remeshing`

Note: The old name `remeshing_isotropic_explicit_remeshing` still works in some versions but `meshing_isotropic_explicit_remeshing` is current.

```python
import pymeshlab as pml

ms = pml.MeshSet()
ms.load_new_mesh("input.ply")  # must be PLY/OBJ (not GLB for PyMeshLab)

# Option 1: percentage-based target edge length
ms.meshing_isotropic_explicit_remeshing(
    iterations=3,
    targetlen=pml.PercentageValue(1)   # target edge = 1% of bbox diagonal
)

# Option 2: absolute target edge length (e.g., 0.5 mm if mesh is in mm)
ms.meshing_isotropic_explicit_remeshing(
    iterations=3,
    targetlen=pml.AbsoluteValue(0.5)   # absolute edge length
)
```

**Key gotcha:** The `adaptive=True` flag sometimes throws `PyMeshLabException: Failed to apply filter` — use `adaptive=False` (default) for safety.

**Workflow when using with trimesh:**

```python
import tempfile, os

def remesh_isotropic(mesh, target_pct=1.0):
    """Remesh using PyMeshLab, return trimesh.Trimesh."""
    ms = pml.MeshSet()
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        tmp_in = f.name
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        tmp_out = f.name

    mesh.export(tmp_in)
    ms.load_new_mesh(tmp_in)
    ms.meshing_isotropic_explicit_remeshing(
        iterations=3,
        targetlen=pml.PercentageValue(target_pct)
    )
    ms.save_current_mesh(tmp_out)
    result = trimesh.load_mesh(tmp_out)
    os.unlink(tmp_in)
    os.unlink(tmp_out)
    return result
```

### 3e. Quadric Edge Collapse Decimation to ~30-50K triangles

**Exact PyMeshLab call:**

```python
def decimate_mesh(mesh, target_faces=40000):
    """Decimate mesh to approximately target_faces using quadric edge collapse."""
    ms = pml.MeshSet()
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        tmp_in = f.name
    mesh.export(tmp_in)
    ms.load_new_mesh(tmp_in)

    # Clean-up before decimation (required for stable QEC)
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()

    # Decimate
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        qualitythr=0.3,          # quality threshold [0-1], higher = better quality
        preservenormal=True,
        optimalplacement=True,
        preserveboundary=True,   # preserve mesh boundary edges
        boundaryweight=1.0,
        planarquadric=False,
    )

    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        tmp_out = f.name
    ms.save_current_mesh(tmp_out)
    result = trimesh.load_mesh(tmp_out)
    os.unlink(tmp_in)
    os.unlink(tmp_out)
    return result
```

**Full parameter list for `meshing_decimation_quadric_edge_collapse`:**
- `targetfacenum` (int): Target number of faces
- `targetperc` (float, 0.0): Target percentage of original faces (0 = use targetfacenum)
- `qualitythr` (float, 0.3): Quality threshold — faces with lower quality are collapsed first
- `preserveboundary` (bool, False): Preserve boundary edges (set True for meshes with borders)
- `boundaryweight` (float, 1.0): Weight for boundary edge preservation
- `optimalplacement` (bool, True): Use optimal vertex placement (slower but better quality)
- `preservenormal` (bool, False): Preserve normals during collapse
- `planarquadric` (bool, False): Add planar constraint to quadrics
- `selected` (bool, False): Apply only to selected faces

**Iterative approach for more accurate target (from PyMeshLab docs):**

```python
ms.meshing_remove_duplicate_vertices()
ms.meshing_remove_unreferenced_vertices()

TARGET = 40000
numFaces = 100 + ms.current_mesh().face_number() - \
           (ms.current_mesh().vertex_number() - TARGET)

while ms.current_mesh().vertex_number() > TARGET:
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=numFaces,
        preservenormal=True
    )
    numFaces -= (ms.current_mesh().vertex_number() - TARGET)
```

### 3f. Re-orient normals

**Using trimesh:**

```python
# Fix winding + normal direction in-place
# Most reliable on watertight meshes
trimesh.repair.fix_normals(mesh, multibody=False)

# If mesh has multiple bodies
trimesh.repair.fix_normals(mesh, multibody=True)

# Also useful — fix just winding
trimesh.repair.fix_winding(mesh)

# Check for broken/inconsistent faces
broken = trimesh.repair.broken_faces(mesh)
print(f"Broken faces: {len(broken)}")
```

**Via PyMeshLab (re-orients all faces consistently):**

```python
ms.meshing_re_orient_faces_coherentely()  # coherent winding across all faces
ms.compute_normal_per_face()
ms.compute_normal_per_vertex()
```

---

## 4. Canonical Orientation via OBB/PCA

### trimesh apply_obb() — what it does

`mesh.apply_obb()` is a method on `trimesh.Trimesh` (inherited from `Geometry3D`). It:
1. Computes the OBB via `trimesh.bounds.oriented_bounds()` (angle-sampling approach, NOT pure PCA)
2. Applies the transform in-place
3. Returns the (4,4) transform matrix that was applied

After `apply_obb()`, the mesh has an **axis-aligned** bounding box, but the axis assignment (which OBB axis → X/Y/Z) is arbitrary.

```python
# oriented_bounds returns (transform_to_origin, extents)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
# to_origin: (4,4) float — matrix that centers OBB at origin
# extents: (3,) float — dimensions of the OBB (NOT sorted)
```

### The full canonical orientation workflow

The goal: longest axis → Y (height), middle axis → X (width), shortest axis → Z (depth).

```python
import trimesh
import numpy as np

def canonical_orient(mesh):
    """
    Orient mesh so that:
      - Center of mass at origin
      - Longest extent → Y axis (height)
      - Middle extent → X axis (width)
      - Shortest extent → Z axis (depth)
    Returns oriented mesh (modified in-place).
    """
    # Step 1: Get OBB transform (moves mesh to OBB frame — AABB at origin)
    transform = mesh.apply_obb()  # returns (4,4) matrix, applies in-place

    # After apply_obb, mesh.bounds gives the AABB
    # The extents tell us the OBB dimensions, but axis order is arbitrary
    extents = mesh.bounding_box.extents  # (3,) — [x_size, y_size, z_size]

    # Step 2: Sort axes: longest→Y (index 1), middle→X (index 0), shortest→Z (index 2)
    # argsort ascending: [smallest, middle, largest]
    order = np.argsort(extents)  # e.g. [2, 0, 1] means z<x<y
    # We want final mapping: longest → Y(1), middle → X(0), shortest → Z(2)
    # final_axis_for_current[i] = where current axis i should go

    # Build permutation: order[2] is currently longest → should be Y (col 1)
    #                    order[1] is currently middle → should be X (col 0)
    #                    order[0] is currently shortest → should be Z (col 2)
    perm = np.zeros(3, dtype=int)
    perm[order[2]] = 1  # longest → Y
    perm[order[1]] = 0  # middle → X
    perm[order[0]] = 2  # shortest → Z

    # Build 4x4 permutation matrix
    P = np.eye(4)
    P[:3, :3] = np.eye(3)[:, perm]  # column permutation

    # Actually, we want to permute: new_coords[i] = old_coords[perm[i]]
    # So we rearrange: vertices[:,0] = old[:,perm[0]], etc.
    # Build correctly as row permutation of identity
    P3 = np.zeros((3, 3))
    for new_ax, old_ax in enumerate(perm):
        P3[new_ax, old_ax] = 1.0
    P4 = np.eye(4)
    P4[:3, :3] = P3

    mesh.apply_transform(P4)

    # Step 3: Resolve sign ambiguity — ensure +Y is "up"
    # Heuristic: the top half of the mesh (Y > centroid_Y) should have
    # more surface area than the bottom (proxy for "top is less flat")
    # For phone-like objects, camera bump is +Z — check if +Z face has
    # higher surface area variation (not easily automated without domain knowledge)
    # Simplest reliable approach: ensure centroid is at origin and
    # positive Y = taller part of object
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)  # re-center after permutation

    # Optional: flip Y if the mass is distributed incorrectly
    # (e.g., if the bottom of the phone is at +Y)
    # This requires domain knowledge — skip in automated pipeline

    return mesh


# Usage
scene = trimesh.load("hunyuan_output.glb", process=False)
mesh = scene.to_mesh()
mesh = canonical_orient(mesh)
print("Extents after orientation:", mesh.bounding_box.extents)
# Should be (width, height, depth) where height > width > depth
```

### Using Open3D's PCA-based OBB (more reliable than trimesh's angle-sampling)

```python
import open3d as o3d
import numpy as np
import trimesh

def get_obb_via_open3d(mesh):
    """
    Use Open3D's PCA-based OBB (more robust than trimesh's angle sampling).
    Returns rotation matrix R and extents.
    """
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    obb = o3d_mesh.get_oriented_bounding_box()
    R = np.asarray(obb.R)        # (3,3) rotation matrix — columns are OBB axes
    center = np.asarray(obb.center)
    extent = np.asarray(obb.extent)  # (3,) dimensions along R's columns
    return R, center, extent

def canonical_orient_open3d(mesh):
    """Canonical orientation using Open3D PCA-based OBB."""
    R, center, extent = get_obb_via_open3d(mesh)

    # Translate to OBB center
    mesh.apply_translation(-center)

    # R columns are OBB axes. Sort by extent: longest→Y, middle→X, shortest→Z
    order = np.argsort(extent)  # ascending: [shortest_idx, middle_idx, longest_idx]

    # New X = old R[:,order[1]], New Y = old R[:,order[2]], New Z = old R[:,order[0]]
    new_R = np.column_stack([R[:, order[1]], R[:, order[2]], R[:, order[0]]])

    # Ensure right-handed coordinate system
    if np.linalg.det(new_R) < 0:
        new_R[:, 2] *= -1  # flip Z to fix handedness

    # Apply rotation: rotate mesh so OBB axes align with world axes
    T = np.eye(4)
    T[:3, :3] = new_R.T  # inverse rotation
    mesh.apply_transform(T)

    return mesh, extent[order]  # returns mesh and (shortest, middle, longest)
```

### Sign ambiguity resolution

The OBB doesn't tell you which end is "top." For a phone:

```python
def resolve_sign_ambiguity(mesh):
    """
    For phone-like objects: ensure +Y faces the screen (flat side).
    Heuristic: the face with higher Z variance = screen side.
    This is approximate — manual inspection may be needed.
    """
    verts = mesh.vertices

    # Check which Z face has more depth variation (camera bump = +Z)
    # Faces with normals pointing -Z are the "back" if camera bump is there
    # This requires domain knowledge; simplest: no-op and document
    pass
```

**Practical note:** For fully automated pipelines, canonical orientation to within a 90-degree rotation is achievable. The remaining sign ambiguity (flip 180°) requires either reference geometry or user confirmation.

---

## 5. Metric Scaling

### Core function

```python
import trimesh
import numpy as np

def scale_mesh_to_dimensions(mesh, width_mm, height_mm, depth_mm, prefer_uniform=False):
    """
    Scale mesh to given real-world dimensions in millimeters.

    Args:
        mesh: trimesh.Trimesh, already canonically oriented
              (X=width, Y=height, Z=depth after canonical_orient)
        width_mm: target width (X extent) in mm
        height_mm: target height (Y extent) in mm
        depth_mm: target depth (Z extent) in mm
        prefer_uniform: if True, scale uniformly to fit within the bounding box
                        defined by width_mm x height_mm x depth_mm (no distortion)
                        if False, apply per-axis non-uniform scaling (exact fit)

    Returns:
        Scaled trimesh.Trimesh (modified in-place)
    """
    # Current extents
    current_extents = mesh.bounding_box.extents  # (width, height, depth) = (X, Y, Z)
    cx, cy, cz = current_extents

    target = np.array([width_mm, height_mm, depth_mm])
    current = np.array([cx, cy, cz])

    if prefer_uniform:
        # Scale uniformly: pick the factor that fits all dimensions within target
        # (no distortion, but mesh may be smaller than target in some axes)
        scale_factors = target / current
        uniform_scale = np.min(scale_factors)  # most constrained axis
        mesh.apply_scale(uniform_scale)
    else:
        # Non-uniform: exact fit to target dimensions (may distort proportions)
        # Use apply_transform with diagonal scale matrix
        scale_matrix = np.diag([
            width_mm / cx,
            height_mm / cy,
            depth_mm / cz,
            1.0
        ])
        mesh.apply_transform(scale_matrix)

    return mesh


# Example: S25 Ultra dimensions
mesh = scale_mesh_to_dimensions(mesh, 77.6, 162.8, 8.2, prefer_uniform=False)
print("Final extents:", mesh.bounding_box.extents)
# Should be close to [77.6, 162.8, 8.2]
```

### When to use uniform vs non-uniform scaling

| Situation | Recommendation |
|---|---|
| Mesh proportions match reference object exactly | Non-uniform (exact fit, `prefer_uniform=False`) |
| Mesh is slightly distorted but topology is clean | Non-uniform (force correct real-world size) |
| Mesh has unknown scale/proportions | Uniform (preserve shape, fit within bounding box) |
| Depth is ambiguous (AI reconstruction artifact) | Fix X,Y non-uniformly; uniform along Z |

### Trimesh scale mechanics

```python
# Uniform scalar
mesh.apply_scale(2.0)           # scales all axes by 2x

# Non-uniform via 4x4 diagonal matrix
S = np.diag([sx, sy, sz, 1.0])
mesh.apply_transform(S)

# Getting current size
extents = mesh.bounding_box.extents   # (3,) — size along X, Y, Z
bounds = mesh.bounds                   # (2, 3) — [[min_x,min_y,min_z],[max_x,max_y,max_z]]
```

**Important:** After `apply_obb()` (which centers at origin), `mesh.bounds` will be `[-e/2, e/2]` along each axis. After scaling to mm, the mesh is centered at origin in mm units.

---

## 6. GLB Export Preserving Textures

### Single mesh export

```python
# Works if mesh.visual is TextureVisuals with PBRMaterial
mesh.export("output.glb")
```

### Scene export (multiple meshes or explicit scene)

```python
# Preferred for preserving scene graph
scene.export("output.glb")

# Or create a new scene from processed mesh
new_scene = trimesh.scene.scene.Scene(geometry={"mesh": mesh})
new_scene.export("output.glb")
```

### The full export_glb signature

```python
trimesh.exchange.gltf.export_glb(
    scene,
    include_normals=None,   # True/False/None (None = auto)
    unitize_normals=True,   # normalize to unit length (GLTF spec requires this)
    tree_postprocessor=None,  # callable to modify GLTF tree before serialization
    buffer_postprocessor=None,
    extension_webp=False,    # use EXT_texture_webp (smaller files, less compat)
    extension_draco=False,   # Draco mesh compression (requires dracox package)
)
```

### Known texture preservation issues

1. **Textured mesh round-trip through PyMeshLab:** PyMeshLab cannot save GLB, so the intermediate format is PLY or OBJ. PLY does not support UV textures in trimesh's loader. Use OBJ for intermediate if preserving UVs:

   ```python
   # For PyMeshLab intermediate, use OBJ (preserves UVs + material)
   mesh.export("intermediate.obj")
   ms.load_new_mesh("intermediate.obj")
   # ... process ...
   ms.save_current_mesh("processed.obj")
   processed = trimesh.load("processed.obj", process=False)
   ```

2. **After pymeshfix:** pymeshfix rebuilds vertex/face arrays. The original UV mapping is destroyed because vertices are renumbered. **You must re-UV after pymeshfix using xatlas.** Store the original texture image before repair.

3. **After decimation:** Same problem — decimation changes mesh topology, breaking UV mapping. Always regenerate UVs with xatlas after any topology-changing operation (pymeshfix, remeshing, decimation).

4. **mesh.export() vs scene.export():** If mesh has a `TextureVisuals`, `mesh.export("output.glb")` creates a scene wrapper and embeds the texture. This works correctly in trimesh 4.x.

5. **OBJ export for textures:**
   ```python
   # Export with texture (creates .obj + .mtl + .png)
   result = mesh.export("output.obj")  # returns dict of {filename: bytes}
   # Write manually:
   with open("output.obj", "wb") as f:
       f.write(result["model.obj"])
   ```

---

## 7. UV Remapping with xatlas After Decimation

### Core concept

After decimation or pymeshfix, the original UV coordinates are invalid (vertices renumbered). xatlas generates a new UV atlas for the new topology. You then need to **bake** the original texture onto the new UV layout.

### xatlas Python API

```python
import xatlas
import numpy as np

# Method 1: Simple parametrize (one mesh)
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
# vmapping: (N_new,) int — maps new vertex index → original vertex index
# indices:  (F, 3) int — face indices into new vertices
# uvs:      (N_new, 2) float — UV coords in [0,1]^2

# Method 2: Atlas class (more control, multiple meshes)
atlas = xatlas.Atlas()
atlas.add_mesh(vertices, faces)

chart_opts = xatlas.ChartOptions()
chart_opts.max_iterations = 3

pack_opts = xatlas.PackOptions()
pack_opts.resolution = 1024   # texture atlas resolution
pack_opts.padding = 2         # pixel padding between charts

atlas.generate(chart_opts, pack_opts)
vmapping, indices, uvs = atlas[0]
```

### Applying new UVs to a trimesh object

```python
def apply_new_uvs(mesh, vmapping, indices, uvs, original_texture_image):
    """
    Apply xatlas-generated UVs to a decimated mesh.
    
    Args:
        mesh: original decimated trimesh.Trimesh
        vmapping: from xatlas.parametrize — new_vert → old_vert index
        indices: new face indices (may differ from mesh.faces!)
        uvs: (N_new, 2) UV coordinates
        original_texture_image: PIL Image (original baked texture)
    
    Returns:
        New trimesh.Trimesh with UV mapping
    """
    from PIL import Image
    import trimesh

    # Remap vertices (xatlas may add seam-duplicate verts)
    new_vertices = mesh.vertices[vmapping]  # shape: (N_new, 3)

    # Build new mesh with xatlas topology
    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=indices,
        process=False
    )

    # Create PBR material with original texture (to be rebaked)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=original_texture_image,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )

    # Apply UV + material
    new_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)

    return new_mesh
```

### Is texture baking needed after decimation?

**Yes, for clean results.** After decimation, you have two options:

**Option A — No rebake (fast, lower quality):**
- Keep the original high-res texture image.
- Generate new UVs with xatlas.
- Assign new UVs + original texture to decimated mesh.
- This works if the mesh topology didn't change too much. Visual artifacts may appear at seams.

**Option B — Full rebake (slower, higher quality):**
- Generate new UVs with xatlas on decimated mesh.
- Render each UV texel's color by ray-casting against the original high-res mesh.
- This is the "correct" approach for large topology changes.
- See HuggingFace blog (vertex-color-to-textured-mesh) for barycentric bake code.

**For the Hunyuan3D pipeline:**
- If decimation is mild (300K → 40K), Option A is usually visually acceptable.
- If the mesh came through pymeshfix (which rebuilds topology completely), Option B is needed.

### Full xatlas + trimesh workflow

```python
import xatlas
import trimesh
import numpy as np
from PIL import Image

def remesh_with_new_uvs(decimated_mesh, original_texture_img, atlas_resolution=2048):
    """
    Generate new UV atlas for a decimated mesh using xatlas.
    Reassigns original texture (no rebake — fast path).
    """
    # Generate UVs
    vmapping, indices, uvs = xatlas.parametrize(
        decimated_mesh.vertices,
        decimated_mesh.faces
    )

    # Remap vertices to xatlas indexing
    new_verts = decimated_mesh.vertices[vmapping]

    # Build new mesh
    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=indices, process=False)

    # Apply UVs + original texture
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        baseColorTexture=original_texture_img,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    new_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)

    return new_mesh
```

---

## 8. What Actually Works in Practice

### Minimum viable cleanup pipeline (tested patterns)

This is the recommended pipeline for CPU-only Colab, balancing quality and reliability:

```python
import trimesh
import pymeshlab as pml
import pymeshfix
import xatlas
import numpy as np
import tempfile, os
from PIL import Image

def process_hunyuan_glb(input_path, output_path,
                         width_mm, height_mm, depth_mm,
                         target_faces=40000):
    """
    Full processing pipeline for Hunyuan3D-2 GLB output.
    """

    # --- STEP 1: Load ---
    scene = trimesh.load(input_path, process=False)
    if isinstance(scene, trimesh.Scene):
        mesh = scene.to_mesh()
    else:
        mesh = scene

    # Preserve original texture image before topology changes
    original_texture = None
    if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'baseColorTexture'):
        original_texture = mesh.visual.material.baseColorTexture

    # --- STEP 2: Remove floaters ---
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        components = sorted(components, key=lambda c: c.area, reverse=True)
        largest_area = components[0].area
        kept = [c for c in components if c.area >= 0.01 * largest_area]
        mesh = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]

    # --- STEP 3: PyMeshLab cleanup (non-manifold, duplicates) ---
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        tmp_clean_in = f.name
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        tmp_clean_out = f.name

    # Export geometry only (no texture — texture is destroyed by topology ops anyway)
    mesh.export(tmp_clean_in)

    ms = pml.MeshSet()
    ms.load_new_mesh(tmp_clean_in)

    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    # Taubin smooth (gentle, preserves shape)
    ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=5)

    # Decimate
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        qualitythr=0.3,
        preservenormal=True,
        optimalplacement=True,
        preserveboundary=False,
    )

    ms.save_current_mesh(tmp_clean_out)
    mesh = trimesh.load_mesh(tmp_clean_out, process=False)
    os.unlink(tmp_clean_in)
    os.unlink(tmp_clean_out)

    # --- STEP 4: Fix normals ---
    trimesh.repair.fix_normals(mesh, multibody=True)

    # --- STEP 5: Canonical orientation ---
    mesh.apply_obb()
    extents = mesh.bounding_box.extents
    order = np.argsort(extents)
    perm = np.zeros(3, dtype=int)
    perm[order[2]] = 1  # longest → Y
    perm[order[1]] = 0  # middle → X
    perm[order[0]] = 2  # shortest → Z
    P3 = np.zeros((3, 3))
    for new_ax, old_ax in enumerate(perm):
        P3[new_ax, old_ax] = 1.0
    P4 = np.eye(4)
    P4[:3, :3] = P3
    mesh.apply_transform(P4)
    mesh.apply_translation(-mesh.centroid)

    # --- STEP 6: Metric scaling ---
    current = mesh.bounding_box.extents
    S = np.diag([width_mm / current[0], height_mm / current[1],
                 depth_mm / current[2], 1.0])
    mesh.apply_transform(S)

    # --- STEP 7: Re-UV with xatlas + re-attach texture ---
    if original_texture is not None:
        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
        new_verts = mesh.vertices[vmapping]
        mesh = trimesh.Trimesh(vertices=new_verts, faces=indices, process=False)
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            baseColorTexture=original_texture,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)

    # --- STEP 8: Export ---
    mesh.export(output_path)
    print(f"Exported to {output_path}")
    print(f"Faces: {len(mesh.faces)}, Vertices: {len(mesh.vertices)}")
    print(f"Extents (mm): {mesh.bounding_box.extents}")

    return mesh
```

### Known bugs and workarounds

| Issue | Symptom | Workaround |
|---|---|---|
| `trimesh.load(..., force='mesh')` drops textures | `mesh.visual` is `ColorVisuals`, no UVs | Use `trimesh.load()` only, never `force='mesh'` for GLB |
| `Scene.dump(concatenate=True)` deprecated | DeprecationWarning, removed in Apr 2025 | Use `scene.to_mesh()` instead |
| `meshing_isotropic_explicit_remeshing(adaptive=True)` crashes | `PyMeshLabException` | Use `adaptive=False` (default) |
| pymeshfix rebuilds vertex array | Original UVs invalid after repair | Save texture image before pymeshfix; re-UV with xatlas after |
| PyMeshLab can't load/save GLB | `IOError` or format not found | Convert via PLY/OBJ intermediates for PyMeshLab steps |
| `trimesh.repair.fix_normals` on non-watertight mesh | Normals partially flipped | Use `multibody=True`; accept partial fix |
| xatlas `parametrize` changes face count | `mesh.faces != indices` | Build a NEW trimesh from `(mesh.vertices[vmapping], indices)` |
| Non-uniform scale distorts normals | Shading artifacts after scale | Re-compute normals after non-uniform scale: `mesh.faces_normals` is cached property, auto-invalidated |
| open3d not available in some Colab sessions | `ModuleNotFoundError` | Use trimesh `apply_obb()` + manual axis sort instead |

### What the minimum viable pipeline is

If you only need a clean, scaled GLB without UV perfection:

1. `trimesh.load(path, process=False)` → Scene → `scene.to_mesh()`
2. Remove floaters via `mesh.split()` + area threshold
3. PyMeshLab: `meshing_repair_non_manifold_edges` + `meshing_repair_non_manifold_vertices` + `meshing_decimation_quadric_edge_collapse`
4. `trimesh.repair.fix_normals(mesh)`
5. `mesh.apply_obb()` + axis permutation
6. Non-uniform scale via `apply_transform(np.diag(...))`
7. `mesh.export("output.glb")`

Steps 3c (Taubin), 3d (isotropic remesh), and 7 (xatlas UV remap) are optional quality improvements. The above 7-step sequence is the minimum that produces a clean, correctly-scaled GLB.

---

## Sources Consulted

- [trimesh 4.12.2 documentation](https://trimesh.org/)
- [trimesh.bounds API](https://trimesh.org/trimesh.bounds.html)
- [trimesh.repair API](https://trimesh.org/trimesh.repair.html)
- [trimesh.exchange.gltf API](https://trimesh.org/trimesh.exchange.gltf.html)
- [trimesh install guide](https://trimesh.org/install.html)
- [GLTF and GLB — DeepWiki trimesh](https://deepwiki.com/mikedh/trimesh/7.1-gltf-and-glb)
- [Loading Objaverse GLBs — trimesh issue #2304](https://github.com/mikedh/trimesh/issues/2304)
- [pymeshfix 0.18.1 documentation](https://pymeshfix.pyvista.org/)
- [pymeshfix API Reference](https://pymeshfix.pyvista.org/api.html)
- [PyMeshLab installation](https://pymeshlab.readthedocs.io/en/latest/installation.html)
- [PyMeshLab filter list](https://pymeshlab.readthedocs.io/en/latest/filter_list.html)
- [PyMeshLab I/O formats](https://pymeshlab.readthedocs.io/en/latest/io_format_list.html)
- [PyMeshLab discussion #318 — target face number](https://github.com/cnr-isti-vclab/PyMeshLab/discussions/318)
- [PyMeshLab issue #85 — isotropic remeshing adaptive flag](https://github.com/cnr-isti-vclab/PyMeshLab/issues/85)
- [xatlas Python bindings — mworchel/xatlas-python](https://github.com/mworchel/xatlas-python)
- [Open3D TriangleMesh API](https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html)
- [HuggingFace — Vertex-Colored to Textured Mesh](https://huggingface.co/blog/vertex-colored-to-textured-mesh)
- [nerf2mesh meshutils.py — real-world PyMeshLab pipeline](https://github.com/ashawkey/nerf2mesh/blob/main/meshutils.py)
- [dreamgaussian4d mesh_utils.py — decimation example](https://github.com/jiawei-ren/dreamgaussian4d/blob/main/mesh_utils.py)
- [Hunyuan3D-2 GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
