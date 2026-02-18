"""
3D Point Cloud Viewer for Gaussian Splatting PLY Files.
This script uses the Viser library to load and interactively render
Gaussian splats exported from the training pipeline.

Features:
- Binary PLY parsing without heavy dependencies.
- Reconstruction of 3D covariances from scales and quaternions.
- Web-based interactive 3D visualization.
"""
import viser
import time
import argparse
import os
import numpy as np

def load_ply(path):
    """
    Parses a Gaussian Splatting PLY file and prepares it for Viser.
    
    The loading process:
    1. Reads the PLY header to identify properties and vertex count.
    2. Parses the binary float data into a coordinate-major array.
    3. Extracts colors (converting SH DC to RGB), opacities, and positions.
    4. Reconstructs 2D/3D covariances from log-scales and quaternions.

    Args:
        path (str): Path to the .ply file.
        
    Returns:
        tuple: (means, covariances, colors, opacities)
    """
    with open(path, "rb") as f:
        # 1. Header Parsing
        line = ""
        header_end = False
        num_vertices = 0
        property_names = []
        
        while not header_end:
            line = f.readline().decode('utf-8').strip()
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property float"):
                property_names.append(line.split()[-1])
            elif line == "end_header":
                header_end = True
        
        # 2. Binary Data Extraction
        # All properties in our exporter are float32
        data = np.fromfile(f, dtype=np.float32)
        
    num_props = len(property_names)
    expected_size = num_vertices * num_props
    
    if data.size != expected_size:
        print(f"Error: Expected {expected_size} floats, got {data.size}")
        return None, None, None, None
        
    data = data.reshape(num_vertices, num_props)
    prop_map = {name: i for i, name in enumerate(property_names)}
    
    # Position: (x, y, z)
    means = data[:, [prop_map['x'], prop_map['y'], prop_map['z']]]
    
    # Opacity
    opacities = data[:, prop_map['opacity']]
    
    # Color: Convert spherical harmonics (DC term) back to RGB [0, 1]
    # The formula is color = f_dc * 0.28209479177387814 + 0.5
    f_dc = data[:, [prop_map['f_dc_0'], prop_map['f_dc_1'], prop_map['f_dc_2']]]
    colors = np.clip(f_dc * 0.28209479177387814 + 0.5, 0.0, 1.0)
    
    # Scales: Gaussians stores log-scales, so we exponentiate them
    scales_data = data[:, [prop_map['scale_0'], prop_map['scale_1'], prop_map['scale_2']]]
    scales = np.exp(scales_data)
    
    # Rotations: Quaternions (w, x, y, z)
    quats = data[:, [prop_map['rot_0'], prop_map['rot_1'], prop_map['rot_2'], prop_map['rot_3']]]
    
    # 3. Covariance Reconstruction
    # The 3D covariance matrix (Σ) is constructed from the rotation matrix (R)
    # and the diagonal scaling matrix (S) as Σ = R S S^T R^T.
    
    # Normalize quaternions to ensure valid rotation matrices
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    q = quats / norms
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Construct rotation matrices from quaternions (N, 3, 3)
    R = np.zeros((num_vertices, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
    # Construct diagonal scaling matrices (N, 3, 3)
    S = np.zeros((num_vertices, 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = scales[:, 0], scales[:, 1], scales[:, 2]
    
    # Compute the covariance matrix: M = R @ S, then Cov = M @ M.T
    M = np.einsum('nij,njk->nik', R, S)
    covs = np.einsum('nij,nkj->nik', M, M)
    
    return means, covs, colors, opacities

def run_ply_viewer(ply_path, port=8080):
    """
    Starts the Viser interactive viewer.
    
    Args:
        ply_path (str): Path to the PLY file to visualize.
        port (int): Port for the web server.
    """
    print(f"Parsing PLY from {ply_path}...")
    means, covs, colors, opacities = load_ply(ply_path)
    
    if means is None:
        return

    print(f"Loaded {len(means)} splats")
    
    # Initialize Viser server
    server = viser.ViserServer(port=port)
    
    # Configure Viser theme (e.g., dark mode)
    server.gui.configure_theme(dark_mode=True)
    
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        print(f"Client connected: {client.client_id}")
        
        # Add Gaussians to the 3D scene
        # Viser expects contiguous arrays for performance
        client.scene.add_gaussian_splats(
            "/gaussians", # Unique name for the splats in the scene
            centers=np.ascontiguousarray(means),
            covariances=np.ascontiguousarray(covs),
            rgbs=np.ascontiguousarray(colors),
            opacities=np.ascontiguousarray(opacities.reshape(-1, 1)), # Reshape to (N, 1) as Viser expects
            visible=True,
        )
        
        # Initial Camera Setup (Optimized for typically centered scenes)
        # Position the camera to look at the origin from a distance
        client.camera.position = (0.0, 0.0, -5.0) # Back up along Z-axis
        client.camera.look_at = (0.0, 0.0, 0.0)   # Look at the origin
        client.camera.up_direction = (0.0, -1.0, 0.0) # Y is down in COLMAP/OpenCV convention
                                                      # (Viser default is Y up)

    print(f"Viewer running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping viewer...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Gaussian Splatting PLY results")
    parser.add_argument("ply_path", type=str, help="Path to the .ply file")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the viewer on")
    args = parser.parse_args()
    
    if not os.path.exists(args.ply_path):
        print(f"Error: File {args.ply_path} not found.")
    else:
        run_ply_viewer(args.ply_path, port=args.port)
