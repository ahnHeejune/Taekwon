# Taekwon

 Taekwon Pose evalation project 

## related project and papers
1. TCMR https://github.com/hongsukchoi/TCMR_RELEASE
2. VIBE https://github.com/mkocabas/VIBE
3. MOT  https://github.com/mkocabas/multi-person-tracker
4. FAST DTW https://github.com/slaypni/fastdtw

To run the TCMR, you need many other packages too.

## code and version 
1. vis_mot_result.py : visualize MOT result (BBox and person id etc) 
2. anal_tcmr.py  : visualize the 3D joint pose projection of TCMR output  
3. posematch.py  : synchronize the two video clips using TCMR output and dtw matching 
4. tcmr_3dvis.py : visualize the TCMR result  in 3D (need chumpy, pyrender, open3d installed)
  
