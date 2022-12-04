import numpy as np
import cv2
import joblib


def check_tcmr_pkl(path):

    # pred_cam: (394, 3)         -  translation 
    # orig_cam:(394, 4)         -  ????
    # verts: (394, 6890, 3)     -  vertices position in 3D  
    # pose: (394, 72)           -  joint angles 24*3   
    # betas: (394, 10)
    # joints3d: (394, 49, 3)    - 3D position of joints (49?)
    # Joints2d:                 -     
    # bboxes: (394, 4)          - cx, cy, width, height
    # frame_ids: (394,)         - frame num in input video 
    
    tcmr_result = joblib.load(path)
    person_ids = list(tcmr_result.keys())

    all_missing = {}
    for person_id in person_ids:
        print(f"person id:{person_id}")
        result = tcmr_result[person_id]
        #print(f"keys:{result.keys()}")
        for key in list(result.keys()):
            #print(f"{key}: {type(result[key])}")
            if result[key] is not None:
                #print(result[key].shape)
                if key == 'frame_ids':
                    frame_ids = result[key]
                    print(f"frame({len(frame_ids)}): {frame_ids.min()} - {frame_ids.max()}")
                    if len(frame_ids) < (frame_ids.max() - frame_ids.min() + 1):
                        #print(f"{frame_ids}")
                        missings = []
                        for i in range(frame_ids.min(), frame_ids.max() +1):    
                            #print(f"{np.where(frame_ids == i)}")
                            if len(np.where(frame_ids == i)[0]) == 0:
                                missings.append(i)
                        print(f"missing:{missings}")    
                    all_missing[person_id] = missing   # add to dictionary                        
                '''
                if key == 'joints3d':
                    j3d = result[key]
                    print(f"{j3d[0,:,:]}")
                    for i in range(len(j3d)):
                        #print(f"{j3d[i,:3,:]}")
                        pass
                '''        
    return all_missing

def load_tcmr_result(path, load_verts = False):

    # load tcmr output file and generate frame to person's local frame mapping 
    # path: tcmr pkl file 
    # return: dictionary with mapping  
    
    # 1. load tcmr pkl 
    tcmr_result = joblib.load(path)
    person_ids = list(tcmr_result.keys())
    
    
    # 2. frame to person ids mapping 
    # 2.1 check the required capacity 
    max_nframe = 0
    num_persons = len(person_ids)
    for person_id in person_ids: 
        frames = tcmr_result[person_id]['frame_ids']
        nframe = frames.max()
        if nframe > max_nframe:
            max_nframe = nframe
   

    # 2.2 mapping global fn (in a video) to person index 
    frame2index  = [[-1 for p in range(num_persons + 1)] for i in range(max_nframe +1)]  
    #print(frame2index)
    for person_id in person_ids: 
        frames = tcmr_result[person_id]['frame_ids']
        for i in range(len(frames)):
            frame2index[frames[i]][person_id] = i    
    #print(f"{frame2index}")
    
    # 3. add to dictionary 
    #tcmr_result['frame2index'] = frame2index
    
    # remove redundants
    if not load_verts:
        for person_id in person_ids: 
            tcmr_result[person_id].pop('verts')
        

    return tcmr_result, frame2index
    

def projection(pred_joints, pred_camera, debug = False):

    ''' 
        modified Pytorch verion after copy  from lib/models/spin.py 
        make 1 batch; No Batch in numpy support and frame-by-frame processing in our case 
        
        3D point to 2D projection with camera 
        pred_joints: nx3 numpy 
        pred_camera:  3d-translation only 

    '''
    
    focal_length = 5000.
    scale_pix    = 224.
    pred_joints = pred_joints.copy()  
    
    pred_joints = np.expand_dims(pred_joints, axis=0)
    batch_size = pred_joints.shape[0]
 
    if debug:
        print(f"pred_joints:{pred_joints[:2,:]}")
        print(f"pred_camera:{pred_camera}")
        
    pred_cam_t = np.array([pred_camera[1], pred_camera[2], 2 * focal_length / (scale_pix * pred_camera[0] + 1e-9)])
    pred_cam_t = np.expand_dims(pred_cam_t, axis=0)
    pred_cam_t = np.repeat(pred_cam_t, batch_size, axis=0)
                 #torch.stack([pred_camera[:, 1],
                 #             pred_camera[:, 2],
                 #             2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    if debug:
        print(f"pred_cam_t:{pred_cam_t[:2,:]}")
        
    camera_center = np.zeros((batch_size, 2))
    rotation = np.eye(3) # no rotation (rotation represented by 0th-SMPL joint angle 
    rotation = np.expand_dims(rotation, axis=0)  # batch index 
    rotation = np.repeat(rotation, batch_size, axis=0) # (1) batch repeat   
   
    if debug:
        print(f"rotation:{rotation[0,:,:]}")
   
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation= rotation, 
                                               translation=pred_cam_t,
                                               focal_length=focal_length,
                                               camera_center=camera_center,
                                               debug = debug)
    if debug:
        print(f"pred_keypoints_2d:{pred_keypoints_2d[0, :5,:]}")
                                              
                                               
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (scale_pix / 2.)
    return pred_keypoints_2d.reshape([-1,2])


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center, debug = False):
    """
    modified Pytorch verion after copy  from lib/models/spin.py 
    make 1 batch; No Batch in numpy support and frame-by-frame processing in our case 
    
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points    
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    N          = points.shape[1]
  
   
    # 1. Extrinsic 
    # 1.1 Rotation 
    #points = np.einsum('bij,bkj->bki', rotation, points)  
    for i in range(batch_size):
        points[i,:,:] = np.einsum('ij,kj->ki', rotation[i,:,:], points[i,:,:])
    if debug:
        print(f"points after rotation:{points[0,:2,:]}")

    # 1.2. Translation 
    translation = np.expand_dims(translation, axis=1)
    points = points + translation  #.unsqueeze(1)
    if debug:
        print(f"translation:{translation[0,:, :]}")
        print(f"points after trnaslation:{points[0,:,:]}")

    # 2.1 projection matrix 
    K = np.zeros([batch_size, 3, 3])
    K[:, 0,0] = focal_length
    K[:, 1,1] = focal_length
    K[:, 2,2] = 1.
    K[:, :-1, -1] = camera_center
    if debug:
        print(f"K:{K[0,:,:]}")

    # 2.2 Apply perspective distortion  ????
    projected_points_homo = points / np.expand_dims(points[:,:,-1], axis=-1)
    if debug:
        print(f"projected_points_homo:{projected_points_homo[0,:2,:]}")
  
    # 2.3 Apply camera intrinsics
    #projected_points = np.einsum('bij,bkj->bki', K, projected_points)
    projected_points  = np.zeros([batch_size,N,3])
    for i in range(batch_size):
        projected_points[i,:,:] = np.einsum('ij,kj->ki', K[i,:,:], projected_points_homo[i,:,:])
    
    if debug:
        print(f"projected_points:{projected_points[0:,:2,:]}")
  
    return projected_points[:, :, :-1]




# http://vision.imar.ro/human3.6m/filebrowser.php
# heejune 1046815
# Map joints to SMPL joints

#############################
# https://github.com/hongsukchoi/TCMR_RELEASE/blob/master/lib/models/smpl.py
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}
JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]

JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
LIBS_ETC  = [ [JOINT_IDS['Nose'], JOINT_IDS['Head (H36M)']],
                [JOINT_IDS['Head (H36M)'], JOINT_IDS['Top of Head (LSP)']],
            [JOINT_IDS['Nose'], JOINT_IDS['Jaw (H36M)']],
            [JOINT_IDS['Nose'], JOINT_IDS['Right Eye']],
            [JOINT_IDS['Right Eye'], JOINT_IDS['Right Ear']],
            [JOINT_IDS['Left Eye'], JOINT_IDS['Left Eye']],
            [JOINT_IDS['Left Eye'], JOINT_IDS['Left Ear']],
            [JOINT_IDS['Nose'], JOINT_IDS['Neck (LSP)']],
            [JOINT_IDS['Neck (LSP)'], JOINT_IDS['Right Shoulder']],
            [JOINT_IDS['Right Shoulder'], JOINT_IDS['Right Elbow']],
            [JOINT_IDS['Right Elbow'], JOINT_IDS['Right Wrist']],
            [JOINT_IDS['Neck (LSP)'], JOINT_IDS['Left Shoulder']],
            [JOINT_IDS['Left Shoulder'], JOINT_IDS['Left Elbow']],
            [JOINT_IDS['Left Elbow'], JOINT_IDS['Left Wrist']],
            [JOINT_IDS['Neck (LSP)'], JOINT_IDS['Spine (H36M)']],
            [JOINT_IDS['Spine (H36M)'], JOINT_IDS['Thorax (MPII)']],
            [JOINT_IDS['Thorax (MPII)'], JOINT_IDS['Pelvis (MPII)']],
            [JOINT_IDS['Pelvis (MPII)'], JOINT_IDS['Left Hip']],
            [JOINT_IDS['Left Hip'], JOINT_IDS['Left Knee']],
            [JOINT_IDS['Left Knee'], JOINT_IDS['Left Ankle']],
            [JOINT_IDS['Pelvis (MPII)'], JOINT_IDS['Right Hip']],
            [JOINT_IDS['Right Hip'], JOINT_IDS['Right Knee']],
            [JOINT_IDS['Right Knee'], JOINT_IDS['Right Ankle']]
            ]
LIBS_OP =  [ [JOINT_IDS['OP Nose'], JOINT_IDS['OP Neck']],
            [JOINT_IDS['OP Nose'], JOINT_IDS['OP REye']],
            [JOINT_IDS['OP REye'], JOINT_IDS['OP REar']],
            [JOINT_IDS['OP Nose'], JOINT_IDS['OP LEye']],
            [JOINT_IDS['OP LEye'], JOINT_IDS['OP LEar']],
            [JOINT_IDS['OP Neck'], JOINT_IDS['OP RShoulder']],
            [JOINT_IDS['OP Neck'], JOINT_IDS['OP LShoulder']],
            [JOINT_IDS['OP LShoulder'], JOINT_IDS['OP LElbow']],
            [JOINT_IDS['OP RShoulder'], JOINT_IDS['OP RElbow']],
            [JOINT_IDS['OP LElbow'], JOINT_IDS['OP LWrist']],
            [JOINT_IDS['OP RElbow'], JOINT_IDS['OP RWrist']],
            [JOINT_IDS['OP Neck'], JOINT_IDS['OP MidHip']],
            [JOINT_IDS['OP MidHip'], JOINT_IDS['OP RHip']],
            [JOINT_IDS['OP RHip'], JOINT_IDS['OP RKnee']],
            [JOINT_IDS['OP RKnee'], JOINT_IDS['OP RAnkle']],
            [JOINT_IDS['OP RAnkle'], JOINT_IDS['OP RBigToe']],
            [JOINT_IDS['OP RAnkle'], JOINT_IDS['OP RSmallToe']],
            [JOINT_IDS['OP RAnkle'], JOINT_IDS['OP RHeel']],
            [JOINT_IDS['OP MidHip'], JOINT_IDS['OP LHip']],
            [JOINT_IDS['OP LHip'], JOINT_IDS['OP LKnee']],
            [JOINT_IDS['OP LKnee'], JOINT_IDS['OP LAnkle']],
            [JOINT_IDS['OP LAnkle'], JOINT_IDS['OP LBigToe']],
            [JOINT_IDS['OP LAnkle'], JOINT_IDS['OP LSmallToe']],
            [JOINT_IDS['OP LAnkle'], JOINT_IDS['OP LHeel']]
            ]
            
            
def visualize_tcmr_result(tcmr_result, frame2index, videofile, fps = 30):
  
    # visualize the results with input 
    # tcmr_result: result ditionary 
    # videofile  : input video 
    # fps: play speed when positive, keyboard driven when negative, 
    #
    global JOINT_NAMES, LIBS_ETC, LIBS_OP
    
    vis_op = True
    vis_etc = False
    
    num_validframe = len(frame2index)
    
    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(videofile) 
    if cap is None:
        print("canot open video file")
        return 
    
    np.set_printoptions(precision=3)
    
    fn = 0 
    while True:
        retval, frame = cap.read() # decode one frame  
        if not retval: # no more frame or error
            break       
        if fn < num_validframe:
            cv2.putText(frame, f"fn={fn}", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,0), 2, cv2.LINE_AA)
            for person_idx in range(len(frame2index[fn])):   
                local_idx = frame2index[fn][person_idx]            
                if local_idx >= 0: 
                    person  = tcmr_result[person_idx]
                    pid_offset = (person_idx-1)
                    #cv2.putText(frame, f"p:{person_idx},betas={person['betas'][local_idx, :2]}, pred_cam={person['pred_cam'][local_idx, :]}", (100,200*person), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,0), 2, cv2.LINE_AA)   
                    cv2.putText(frame, f"pid:{person_idx}", (50,50 + 150*pid_offset + 0), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,0,0), 2, cv2.LINE_AA) 
                    # shape and pose 
                    cv2.putText(frame, f"betas={person['betas'][local_idx, :2]} ...", (100,50 + 150*pid_offset + 0), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,0,0), 1, cv2.LINE_AA) 
                    cv2.putText(frame, f"pose={person['pose'][local_idx, :3]} ...", (100,50 + 150*pid_offset + 25*1), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,0,0), 1, cv2.LINE_AA)
                    
                    # 2. bbox
                    bbox = person['bboxes'][local_idx, :]
                    cx, cy = bbox[0:2]
                    width, height = bbox[2:]
                    cv2.rectangle(frame, (int(cx-width/2), int(cy-height/2)), (int(cx + width/2), int(cy + height/2)), (255,255,0)) 
                    
                    # 3. camera and 3d joint 
                    cv2.putText(frame, f"pred_cam={person['pred_cam'][local_idx, :]}", (100,50 + 150*pid_offset + 25*2), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"orig_cam={person['orig_cam'][local_idx, :]}", (100,50 + 150*pid_offset + 25*3), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"joints3d={person['joints3d'][local_idx, 0, :]}...", (100,50 + 150*pid_offset + 25*4), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,0,0), 1, cv2.LINE_AA)
                    # project 3d joint to 2d 
                    j2d = projection(person['joints3d'][local_idx], person['pred_cam'][local_idx], debug = False)
                    #print(f"j2d:{j2d[:2,:]}")
                    j2d[:,0] = j2d[:,0]*width/2  + cx
                    j2d[:,1] = j2d[:,1]*height/2 + cy
                    #print(f"j2d ({len(j2d)}:{j2d[:2,:]}")
                    # draw projected joints 
                    for i in range(len(j2d)):
                        color = (255, 0, 0) if i <= 24 else (0, 0, 255)    ## OP or not 
                        if vis_op and i <= 24:          
                            cv2.circle(frame, (int(j2d[i,0]),int(j2d[i,1])) , radius = 5, color =color, thickness = -1)
                            #cv2.putText(frame, f"{JOINT_NAMES[i]}", (int(j2d[i,0]),int(j2d[i,1])), cv2.FONT_HERSHEY_SIMPLEX,  0.4, color, 1, cv2.LINE_AA) 
                        elif vis_etc and i > 24:
                            cv2.circle(frame, (int(j2d[i,0]),int(j2d[i,1])) , radius = 5, color =color, thickness = -1)
                        '''
                        if i <= 24:
                            print(f"OP points: {(int(j2d[i,0]),int(j2d[i,1]))}")
                        else:
                            print(f"non-OP pt: {(int(j2d[i,0]),int(j2d[i,1]))}")
                        '''
                    # @TODO : draw lines for limbs
                    if vis_op:
                        for limb in LIBS_OP:
                            part1, part2 = limb
                            cv2.line(frame, (int(j2d[part1,0]),int(j2d[part1,1])), (int(j2d[part2,0]),int(j2d[part2,1])),(255,0,0), 2)
                    
                    # @TODO : draw lines for limbs
                    if vis_etc:
                        for limb in LIBS_ETC:
                            part1, part2 = limb
                            cv2.line(frame, (int(j2d[part1,0]),int(j2d[part1,1])), (int(j2d[part2,0]),int(j2d[part2,1])),(0,0,255),2)
               
                    
    
        cv2.imshow('Display', frame)
        if fps < 0:
            key  = cv2.waitKey(-1) # in ms
        else:
            key = cv2.waitKey((1000 + fps//2)//fps) # in ms
        if key ==27:    # Esc key to stop
            break       
       
        fn +=1
    cv2.destroyAllWindows()
    if cap.isOpened():
        cap.release()
 

def check_extra_joint():

    ''' 
        regression 
        9 extra joint from SMPL vertices
        
    '''
    import os
    jextra_regressor = np.load(os.path.join('base_data', 'base_data', 'J_regressor_extra.npy'))
    print(f"jextra_regressor:{jextra_regressor.shape}")  # 9x6890
    
 
 
 
    
if __name__ == "__main__":

    import sys


    if len(sys.argv) < 2:
        print(f"usage: python {sys.argv[0]} tcmr_pkl <tmcr_input_video> <fps>")
        exit()

    elif len(sys.argv) < 3:
    
        check_tcmr_pkl(sys.argv[1])
        #load_tcmr_result(sys.argv[1])
    else:    
        pklfile = sys.argv[1]
        tcmr_result, frame2index = load_tcmr_result(pklfile)
        videofile = sys.argv[2] # pklfile.replace('pkl', 'mp4').replace('output', 'input')
        if len(sys.argv) > 3:
            visualize_tcmr_result(tcmr_result, frame2index, videofile, int(sys.argv[3]))
        else:
            visualize_tcmr_result(tcmr_result, frame2index, videofile)
        

