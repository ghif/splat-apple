import struct
import numpy as np

f_path = '/Users/mghifary/Work/Code/AI/data/gsplat/nerf_real_360/pinecone/sparse/0/points3D.bin'
with open(f_path, 'rb') as f:
    num_points = struct.unpack('<Q', f.read(8))[0]
    print(f'Num points: {num_points}')
    for i in range(5):
        data = f.read(43)
        p = struct.unpack('<QdddBBBd', data)
        print(f'Point {i}: ID={p[0]}, XYZ={p[1:4]}, RGB={p[4:7]}, Err={p[7]}')
