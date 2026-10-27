# Lumber_Segmentation
This project segments the five lumbar vertebrae (L1-L5) from 3D CT scans and computes vertebral body centroids in RAS world coordinates.  The model is a four-level 3D U-Net trained on complete lumbar annotations from TotalSegmentator v3. It is evaluated on a held-out TotalSegmentator test split and on VerSe 2019 as an out-of-domain dataset.
