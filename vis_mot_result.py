import joblib
import numpy as np
import cv2

''' Analysis MOT part  '''



def check_tcmr_pkl(path):
 
    '''
        check the missing frames for each person_id 
       
    '''
    
    tcmr_result=joblib.load(path)
    person_ids=list(tcmr_result.keys())

    all_missing = {}

    for person_id in person_ids:
        print(f"person id: {person_id}")
        result=tcmr_result[person_id]
        missing = []
        for key in list(result.keys()):
            if result[key] is not None:
                print(f" {key}: {result[key].shape}")
                if key=='frame_ids':
                    frame_ids=result[key]
                    print(f"{len(frame_ids)} frames: from {frame_ids.min()} to {frame_ids.max()}")
                    count_missing=0
                    if len(frame_ids)<(frame_ids.max()-frame_ids.min()):
                        for i in range(frame_ids.min(),frame_ids.max()+1):
                            if len(np.where(frame_ids==i)[0])==0:
                                missing.append(i)
                                count_missing+=1
                        print(f"{count_missing} missing: {missing}")

        all_missing[person_id] = missing   # add to dictionary

    return all_missing

def load_tcmr_result(path):
    
    ''' load tcmr result plk file 
        then make a list frames to index of tcmr_result[person_id][index]  
    '''

    #1.load tcmr.pkl
    tcmr_result=joblib.load(path)
    person_ids=list(tcmr_result.keys())

    print(f"tcmr_result.keys():{tcmr_result.keys()}")

    #2.frame to person id mapping
    #2.1.check
    max_nframe=0
    num_person=len(person_ids)
    for person in person_ids:
        frames=tcmr_result[person]['frame_ids']
        # print(frames)
        nframe=frames.max()  #=len(frames)
        print(f"range of frames: {frames.min()}- {frames.max()}, #:{frames.shape[0]}")
        if nframe>max_nframe:
            max_nframe=nframe

    #2.2.mapping global to person index
    frame2index=[[-1 for p in range(num_person+1)] for i in range(max_nframe+1)] #2-dim array =-1
    for person_id in person_ids:
        frames=tcmr_result[person_id]['frame_ids']
        for i in range(len(frames)):
            frame2index[frames[i]][person_id]=i
    # print(frame2index)

    #3. add to dictionary
    #tcmr_result['frame2index']=frame2index

    #remove redundants
    for person_id in person_ids:
        tcmr_result[person_id].pop('verts')

    return tcmr_result, frame2index

def show_bbox(tcmr_result, frame2index, videofile, fps=30):

    num_validframe = len(frame2index)
    person_ids = list(tcmr_result.keys())
    print(f"personids:{person_ids}")

    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(videofile)
    if cap is None:
        print("canot open video file")
        return

    mis=[]
    fn = 0


    npersons = len(person_ids)
    prev_bbs = {}
    for person_id in person_ids:
        bb = np.zeros(4)
        prev_bbs[person_id] = bb

    while fn  in range(num_validframe):     # for each frame

        # print(fn)

        retval, frame = cap.read()  # decode one frame
        if not retval:  # no more frame or error
            break

        for person_idx in person_ids:  # for each person

            person_idx = int(person_idx)
            local_idx = frame2index[fn][person_idx]
            #print(f"local_idx: {local_idx}")

            #if fn in all_missings[person]:
            if local_idx == -1:
                    cv2.putText(frame, f"fn={fn}, pid={person_idx} missing", (0, person_idx*25), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                    bbox = prev_bbs[person_idx]
                    bb_color = (0, 0, 255)  # RED
            else:
                    cv2.putText(frame, f"fn={fn},  pid={person_idx} detected", (0, person_idx*25), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                    person = tcmr_result[person_idx]
                    # bbox
                    bbox = person['bboxes'][local_idx, :]
                    prev_bbs[person_idx]   = bbox
                    bb_color = (0, 255, 0)  # GREEN

            #print(f"fn: {fn}, box: {bbox}")
            cx, cy = bbox[0:2]
            width, height = bbox[2:]
            cv2.rectangle(frame, (int(cx - width / 2), int(cy - height / 2)),
                          (int(cx + width / 2), int(cy + height / 2)), bb_color,3)

        cv2.imshow('Display', frame)

        if fps < 0:
            key = cv2.waitKey(-1)  # in ms
        else:
            key = cv2.waitKey((1000 + fps // 2) // fps)  # in ms

        if key == 32:
            cv2.waitKey()
        if key == 27:  # Esc key to stop
            return

        fn += 1


    cv2.destroyAllWindows()
    if cap.isOpened():
        cap.release()

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 3:
        print(f"usage: python {sys.argv[0]} <tcmr pkl> <tcmr input video> [fps]")
        exit()
    pkl=sys.argv[1]
    videofile = sys.argv[2]
    fps = 30
    if len(sys.argv) > 3:
        fps = int(sys.argv[3])

    #all_missings = check_tcmr_pkl(pkl)   # @TODO make both in one time
    
    tcmr_result, frame2index = load_tcmr_result(pkl) #
    show_bbox(tcmr_result, frame2index, videofile, fps=fps)


