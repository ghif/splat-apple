
"""
Simple script to visualize randomly generated 3D Gaussians using the Viser library. 
It demonstrates how to compute 3D covariances from scales and quaternions and 
render them in a real-time interactive 3D viewer.

"""
import viser
import time
import argparse
import numpy as np

def compute_covariances(scales, quats):
    """
    Compute 3D covariances from scales and quaternions.
    Σ = R S S^T R^T
    """
    n = scales.shape[0]
    
    # R from quaternion
    # Normalize quaternions
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    R = np.zeros((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
    # S from scale (scales are logged in Gaussian Splatting)
    s = np.exp(scales)
    S = np.zeros((n, 3, 3))
    S[:, 0, 0] = s[:, 0]
    S[:, 1, 1] = s[:, 1]
    S[:, 2, 2] = s[:, 2]
    
    # M = R @ S
    M = np.einsum('nij,njk->nik', R, S)
    
    # Cov = M @ M.T
    covs = np.einsum('nij,nkj->nik', M, M)
    
    return covs

def generate_random_gaussians(num_gaussians=1000):
    """Generate random Gaussian data for Viser."""
    # Means: Gaussian distribution around center
    means = np.random.normal(0, 1.0, (num_gaussians, 3))
    
    # Scales: Log scales
    # -3 to -1 result in exp(-3) ~ 0.05 to exp(-1) ~ 0.36
    scales = np.random.uniform(-3.5, -1.5, (num_gaussians, 3))
    
    # Rotations: Random quaternions
    quats = np.random.randn(num_gaussians, 4)
    
    # Colors: Random RGB
    colors = np.random.uniform(0, 1, (num_gaussians, 3))
    
    # Opacities: Random highish opacity
    opacities = np.random.uniform(0.1, 0.8, (num_gaussians, 1))
    
    covs = compute_covariances(scales, quats)
    
    return means, covs, colors, opacities

def run_viewer(num_gaussians):
    server = viser.ViserServer()
    server.configure_theme(dark_mode=True)
    
    print(f"Generating {num_gaussians} random Gaussians...")
    means, covs, colors, opacities = generate_random_gaussians(num_gaussians)
    
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        print(f"Client connected: {client.client_id}")
        
        # Add the splats
        client.scene.add_gaussian_splats(
            "/gaussians",
            centers=np.ascontiguousarray(means),
            covariances=np.ascontiguousarray(covs),
            rgbs=np.ascontiguousarray(colors),
            opacities=np.ascontiguousarray(opacities),
            visible=True,
        )
        
        # Initial camera setup
        client.camera.position = (0.0, 0.0, -5.0)
        client.camera.look_at = (0.0, 0.0, 0.0)
        client.camera.up_direction = (0.0, -1.0, 0.0)

    # Add a button to regenerate
    with server.add_gui_folder("Controls"):
        num_slider = server.add_gui_slider("Step count", min=100, max=10000, step=100, initial_value=num_gaussians)
        regen_button = server.add_gui_button("Regenerate")

    @regen_button.on_click
    def _(_):
        n = num_slider.value
        print(f"Regenerating {n} Gaussians...")
        new_means, new_covs, new_colors, new_opacities = generate_random_gaussians(n)
        
        # Update for all clients
        server.scene.add_gaussian_splats(
            "/gaussians",
            centers=np.ascontiguousarray(new_means),
            covariances=np.ascontiguousarray(new_covs),
            rgbs=np.ascontiguousarray(new_colors),
            opacities=np.ascontiguousarray(new_opacities),
            visible=True,
        )

    print("Viewer running... Press Ctrl+C to stop.")
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=2000, help="Number of Gaussians to generate")
    args = parser.parse_args()
    
    run_viewer(args.num)
