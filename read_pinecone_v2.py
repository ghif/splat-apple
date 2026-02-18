import struct
import numpy as np

f_path = '/Users/mghifary/Work/Code/AI/data/gsplat/nerf_real_360/pinecone/sparse/0/points3D.bin'
with open(f_path, 'rb') as f:
    num_points = struct.unpack('<Q', f.read(8))[0]
    print(f'Num points: {num_points}')
    for i in range(10):
        data = f.read(43)
        p = struct.unpack('<QdddBBBd', data)
        track_length = struct.unpack('<Q', f.read(8))[0]
        f.read(track_length * 8)
        print(f'Point {i}: ID={p[0]}, RGB={p[4:7]}, XYZ={p[1:4]}, TrackLen={track_length}')
