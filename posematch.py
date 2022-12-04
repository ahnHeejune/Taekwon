import numpy as np
import cv2
import joblib


''' Load pose Sequence '''

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

    missings = []  ##### TODO
   
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
                        #missings = []
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

def load_tcmr_result(path, fill_missing = False):

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
    
    if fill_missing:
        frame_ids = tcmr_result[1]['frame_ids']
        poses = tcmr_result[1]['pose'].reshape((-1,24, 3))
        poses_new = np.zeros([frame_ids.max()+1, 24,3])
        for i in range(frame_ids.max()+1):    
            #print(f"{np.where(frame_ids == i)}")
            match_frame_ids = np.where(frame_ids == i)[0]
            #print(f"match_frame_ids:{match_frame_ids}, {len(match_frame_ids)} for {i}")
            if len(match_frame_ids) == 0:
                if i != 0:
                    poses_new[i, :, :] = poses_new[i-1, :, :]
                else:
                    poses_new[i, :, :] = 0
            else:
                poses_new[i, :, :] = poses[match_frame_ids[0], :, :]   
                
        tcmr_result[1]['pose'] = poses_new.reshape([-1,24*3])    # update with filled pose    
    
    # remove redundants
    for person_id in person_ids: 
        tcmr_result[person_id].pop('verts')
    
    return tcmr_result, frame2index
 
 
''' Visualization  '''
 
def crop_bbox(frame, bbox): 

    # x_max, y_max:  
    
    y_max, x_max = frame.shape[:2] 
    
    cx, cy = int(bbox[0]), int(bbox[1])
    width, height = int(bbox[2]), int(bbox[3])
    sy, ey, sx, ex = cy-height//2, cy + height//2, cx - width//2, cx + width//2

    if sx < 0:
        sx1 = -sx
        sx  = 0
    else:
        sx1 = 0
        
    if sy < 0:
        sy1 = -sy
        sy  = 0
    else:
        sy1 = 0
        
    if ex > x_max:
        ex = x_max
    ex1 = sx1 + (ex - sx)
    
    if ey > y_max:
        ey = y_max
    ey1 = sy1 + (ey - sy)
    
    croped = np.zeros((height, width, 3), dtype = np.uint8)   
    croped[sy1:ey1, sx1:ex1, :] = frame[sy:ey, sx:ex ,:] 
    
    return croped
    

           
def visualize_comparison(vid_ref, vid_tgt, result_ref, result_tgt, matches, costmat = None, fps = 30, save = False):
  
    # visualize the results with input 
    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
    cap_ref = cv2.VideoCapture(vid_ref) 
    cap_tgt = cv2.VideoCapture(vid_tgt)    
    if cap_ref is None:
        print("canot open ref video file")
        return 
    if cap_tgt is None:
        print("canot open target video file")
        return     
        
    frame_width = 640
    frame_height = 640
    if save:
        out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width*2,frame_height))
    
    person_ref = result_ref[1]
    person_tgt = result_tgt[1]

  
    fn_r = -1
    fn_t = -1
    for fn_r_new, fn_t_new in matches:
        
        while fn_r < fn_r_new or fn_t < fn_t_new:
        
            if fn_r  < fn_r_new: 
                retval, frame_ref = cap_ref.read() # decode one frame  
                fn_r +=1
                if not retval: # no more frame or error
                    break       
                if fn_r >= len(person_ref['bboxes']):
                    break           
                # take only BBOX area and resize of needed 
                bbox_ref = person_ref['bboxes'][fn_r, :]
                frame_ref = crop_bbox(frame_ref, bbox_ref)  
                frame_ref = cv2.resize(frame_ref, dsize =(frame_width, frame_height))
                      
            if fn_t < fn_t_new: 
                retval, frame_tgt = cap_tgt.read() # decode one frame  
                fn_t +=1
                if not retval: # no more frame or error
                    break      
                if fn_t >= len(person_tgt['bboxes']):
                    break         
                bbox_tgt = person_tgt['bboxes'][fn_t, :]
                frame_tgt = crop_bbox(frame_tgt, bbox_tgt)
                # resize (assume BB's aspect ration is 1:1
                frame_tgt = cv2.resize(frame_tgt, dsize =(frame_width, frame_height))
                          
            frame = np.hstack((frame_ref, frame_tgt))
            cv2.putText(frame, f"fn={fn_r}", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"fn={fn_t}", (frame_width,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            if costmat is not None:
                cv2.putText(frame, f"acc={1-costmat[fn_r, fn_t]:.2f}", (frame_width - 200,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            
           
            cv2.imshow('Display', frame)
            
            if save:        
                out.write(frame)

            if fps < 0:
                key  = cv2.waitKey(-1) # in ms
            else:
                key = cv2.waitKey((1000 + fps//2)//fps) # in ms
            if key == 27:    # Esc key to stop
                return       
         
        if key == 27:    # Esc key to stop
                return  
                
    cv2.destroyAllWindows()
    if cap_ref.isOpened():
        cap_ref.release() 
    if cap_tgt.isOpened():
        cap_tgt.release()        

    
    if save and out.isOpened():
        out.release()
         
 
''' Pose Distance ''' 

debug = False
count_dist = 0 
pose_weights = np.ones(24)  # weighting 
pose_weights[0] = 3
pose_weights_total = np.sum(pose_weights)
 
def calculate_pose_dist(pose1, pose2):
    
    global debug, count_dist, pose_weights, pose_weights_total
    
    # pose1, pose2: numpy 24x3
    # return sum of cosine similairities between two rodrgues vectors  
    # An Euler-Rodrigues vector represents a rotation by integrating a direction cosine of a rotation axis with the tangent of half the rotation angle
    # 
   
    from numpy import dot
    from numpy.linalg import norm
    
    if debug:
        #print(f"dist:{pose1[0,0]} to {pose2[0,0]}")
        count_dist +=1
    
    if False:
        sum_cossim1 = 0
        for i in range(0, 24):  # except the global one 
            sum_cossim1 += pose_weights[i] * dot(pose1[i], pose2[i])/(norm(pose1[i])*norm(pose2[i]))    
        sum_cossim1 = sum_cossim/pose_weights_total # 24 # 23.0 
    else:
        debug = False
        if debug:
            print(f"pose1:{pose1}")
            print(f"pose2:{pose2}")
        norm1 = np.sqrt(np.sum(np.square(pose1), axis = 1))
        if debug:
            print(f"norm1:{norm1}")
        norm2 = np.sqrt(np.sum(np.square(pose2), axis = 1))
        if debug:
            print(f"norm1:{norm2}")
        dot   = np.einsum('ij,ij->i',pose1, pose2)
        if debug:
            print(f"dot:{dot}")
        sum_cossim = dot/norm1/norm2 
        if debug:
            print(f"sum_cossim:{sum_cossim}")
        sum_cossim = np.sum(sum_cossim)/ pose_weights_total
        if debug:
            print(f"sum_cossim: {sum_cossim} vs {sum_cossim1}")
            #_ = input()
    
    if 1-sum_cossim < 0:
        return 0
    else:
        return 1-sum_cossim


''' FAST Sequence Matching '''

def match_poseseq_fast(seq_r, seq_t, debug = False):

    #global debug
    ''' 
        baseline implementation: simple matching the same frame 
        
        seq_ref: tcmr output 
        seq_tgt: tcmr_output
        return  ref2tgt frame, cost 
        
        @TODO implement the matching algorithm in your own way 
              First candiate is Dynamic time warping using pose-angle(?) distance
              Note that the first angle (global angle) should not be used because it is in fact camera to human direction) 
    
    '''

    #nframe_r = len(seq_r[1]['frame_ids'])
    #nframe_t = len(seq_t[1]['frame_ids'])
  
    # SMPL pose are in Rodrigues  
    poses_r = seq_r[1]['pose'].reshape((-1,24, 3))
    poses_t = seq_t[1]['pose'].reshape((-1,24, 3))
    nframe_r = len(poses_r)
    nframe_t = len(poses_t)
    if debug:
        print(f"poses_r({poses_r.shape}):{poses_r[0, :5]}")
        print(f"poses_t({poses_t.shape}):{poses_t[0, :5]}")  

    
    if debug:
        poses_r[:,0,0] = np.arange(len(poses_r))
        poses_t[:,0,0] = np.arange(len(poses_t))
        
    from fastdtw import fastdtw
    distance, path = fastdtw(poses_r, poses_t, radius = 1, dist=calculate_pose_dist)
    
    if debug:
        print(f"count_dist: {count_dist}/{nframe_r*nframe_t}")
        print(f"distance: {distance}")
        print(f"initial path: {path[:20]}")
        
    
    # cost calcuation only for path 
    costmat = np.zeros([nframe_r, nframe_t])    
    costmat[:,:] = np.inf
    if debug:
        print("Calculating cost matrix....")
    for (r, t) in path:
            pose_r = poses_r[r] 
            pose_t = poses_t[t]
            costmat[r,t] = calculate_pose_dist(pose_r, pose_t)
    
    #exit()
    return path, costmat, None
    
  
    
''' EXACT VERSION '''

    
def dtw_exact(costmat, poses_r, poses_t, window):

    '''
     from "https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd"
     
    
    1. Every index from the first sequence must be matched with one or more indices from the other sequence and vice versa
    2. The first index from the first sequence must be matched with the first index from the other sequence  => Not for us!
      (but it does not have to be its only match)
    3. The last index from the first sequence must be matched with the last index from the other sequence    ==> Not for us!
       (but it does not have to be its only match)
    4. The mapping of the indices from the first sequence to indices from the other sequence must be monotonically increasing, and vice versa,
       i.e. if j > i are indices from the first sequence, then there must not be two indices l > k in the other sequence, such that index i is matched with index l and index j is matched with index k , and vice versa 
       
    '''

    n, m = len(poses_r), len(poses_t)
    w = np.max([window, abs(n-m)])
    
    dtw_mat = np.zeros((n+1, m+1))
    #dtw_mat[:, :] = np.inf
    for i in range(n+1):
        for j in range(m+1):
            dtw_mat[i, j] = np.inf
    
    dtw_mat[0, 0] = 0       
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            dtw_mat[i, j] = 0
    print("Calculating dtw_mat....")
    for r in range(1, n+1):
        if r%100 == 1:
            print(f"running for r:{r-1}/{n}")
        for t in range(np.max([1, r-w]), np.min([m, r+w])+1):
            cost =  costmat[r-1,t-1] 
            # take last min from a square box ??
            last_min = np.min([dtw_mat[r-1, t], dtw_mat[r, t-1], dtw_mat[r-1, t-1]])
            dtw_mat[r, t] = cost + last_min
            
    return dtw_mat
    

def test_pose_dist(costmat, poses_r, poses_t):

    #n, m = len(poses_r), len(poses_t)
    n = 10
    m = 10

    for r in range(1, n+1):
        for t in range(1, m+1):
            pose_r = poses_r[r-1] 
            pose_t = poses_t[t-1]
            costmat[r-1,t-1] = calculate_pose_dist(pose_r, pose_t)
            #print(f"{r-1},{t-1}: {costmat[r-1,t-1]}")
    
    print(f"costmat:{costmat[:10,:10]}")  
    
  
#def match_poseseq_exact(seq_r, seq_t, missing_frame_r, missing_frame_t, debug = False):
def match_poseseq_exact(seq_r, seq_t, debug = False):

    ''' 
        exact implemetation using dynamic programming method 
        
        min_accum_cost(r, t) =  cost(r,t) + min( min_accum_cost(r-1, t-1),  min_accum_cost(r, t-1), min_accum_cost(r-1, t)) 
        
        seq_ref: tcmr output 
        seq_tgt: tcmr_output
        return  ref2tgt frame, cost 
    '''

    #nframe_r = len(seq_r[1]['frame_ids'])
    #nframe_t = len(seq_t[1]['frame_ids'])
  
    # SMPL pose are in Rodrigues  
    poses_r = seq_r[1]['pose'].reshape((-1,24, 3))
    poses_t = seq_t[1]['pose'].reshape((-1,24, 3))
    
    if debug:
        print(f"poses_r({poses_r.shape}):{poses_r[0, :5]}")
        print(f"poses_t({poses_t.shape}):{poses_t[0, :5]}")  

    nframe_r = len(poses_r)
    nframe_t = len(poses_t)
  
    
    # DTW method for matching 
    
    costmat = np.zeros([nframe_r, nframe_t])    
    costmat[:,:] = np.inf
    #### @TODO BANGGGGGGG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("Calculating cost matrix....")
    for r in range(len(poses_r)):
        if r%100 == 1:
            print(f"running for r:{r-1}/{nframe_r}")
        for t in range(len(poses_t)):
            pose_r = poses_r[r] 
            pose_t = poses_t[t]
            costmat[r,t] = calculate_pose_dist(pose_r, pose_t)
    
    dtw_mat = dtw_exact(costmat, poses_r, poses_t, window = max(len(poses_r)-1, len(poses_t)-1))
    #print(f"costmat:{costmat[:10,:10]}")
    #print(f"dtw_mat:{dtw_mat[:10,:10]}")
    
    # 3. extract the minimum path 
    #    backward searching 
    # 3.2 the global min path exit postion  
    r = nframe_r -1
    t = nframe_t -1
   
    min_ind_t = np.argmin(dtw_mat[-1, :])
    min_t = dtw_mat[-1, min_ind_t]
    if debug:
        print(f"min_t: {min_t} at {min_ind_t}")
        #print(f"dtw_mat[-2, :]: {dtw_mat[-2, 750:]}")
        #print(f"dtw_mat[-1, :]: {dtw_mat[-1, 750:]}")
    t = min_ind_t  
  
    min_ind_r = np.argmin(dtw_mat[:, -1])
    min_r = dtw_mat[min_ind_r, -1]
    if debug:
        print(f"min_r: {min_r} at {min_ind_r}")
    
    '''
    #### @TODO: DO we have full Matrix filled or not?
    if min_t < min_r:
        t = min_ind_t
    else:
        r = min_ind_r
    '''
  
  
    #missing_r = len(missing_frame_r)
    #missing_t = len(missing_frame_t)
    minpath = [ (r, t) ]
    while r>= 0 and t>=0:
        min_at = np.argmin([dtw_mat[r-1,t], dtw_mat[r,t-1], dtw_mat[r-1,t-1]])
        if min_at == 0:
            r -= 1
        elif min_at == 1:
            t -= 1
        else:
            r -= 1
            t -= 1
            
        if  r < 0 or t < 0:
            break;
        else:
            #if r in missing_frames_r:
            #    missing_r -=1 
            #if t in missing_frames_t:
            #    missing_t -=1 
            #minpath.insert(0, (r + missing_r, t + missing_t) ) 
            minpath.insert(0, (r, t) ) 
       
    if debug:
        #print(f"path: {minpath}")    
        img_dtw_mat = np.array(dtw_mat)
        for (r, t) in minpath:
            img_dtw_mat[r,t] = 0  # draw a minimal path 
        import matplotlib.pyplot as plt
        plt.imshow(img_dtw_mat), plt.xlabel('t'), plt.ylabel('r')
        plt.show()
        
    
    return minpath, costmat, dtw_mat    


if __name__ == "__main__":

    ''' 
        currently, only single person file supports  (@TODO segmented persons)  
    '''
    import sys

    if len(sys.argv) < 3:
        print(f"usage: python {sys.argv[0]} ref_tcmr_pkl tgt_tcmr_pkl <fps>")
        exit()

    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    np.set_printoptions(precision=2)

    pkl_ref = sys.argv[1]
    pkl_tgt = sys.argv[2]
    seq_ref, frame2index = load_tcmr_result(pkl_ref, fill_missing = True)
    seq_tgt, frame2index = load_tcmr_result(pkl_tgt, fill_missing = True)
    
    
    # check
    if False:     
        missing_frames_r = check_tcmr_pkl(pkl_ref)
        missing_frames_t = check_tcmr_pkl(pkl_tgt)
        #exit()
    
    
    vid_ref  = pkl_ref.replace("pkl", "mp4")
    vid_tgt  = pkl_tgt.replace("pkl", "mp4")
    
    if False:
        frame_matches, costmat, _ = match_poseseq_exact(seq_ref, seq_tgt, debug = True)
        #frame_matches = match_poseseq_exact(seq_ref, seq_tgt, missing_frames_r, missing_frames_t, debug = True)
        visualize_comparison(vid_ref, vid_tgt, seq_ref, seq_tgt, frame_matches, costmat = costmat, fps= fps, save = True)
    else:    
        frame_matches, costmat, _ = match_poseseq_fast(seq_ref, seq_tgt)    
        visualize_comparison(vid_ref, vid_tgt, seq_ref, seq_tgt, frame_matches, costmat = costmat, fps= fps, save = True)
    
   
      
  
    
    