import numpy as np
import plyfile

scannet_labels_path = "/home/knuvi/Desktop/song/data/scannet_sample/train/scene0027_02/scene0027_02_vh_clean_2.labels.ply"

# ScanNet 레이블 확인
with open(scannet_labels_path, "rb") as f:
    plydata = plyfile.PlyData.read(f)
    scannet_labels = np.asarray(plydata['vertex']['label'])

print(f"Unique labels in ScanNet labels: {np.unique(scannet_labels)}")
