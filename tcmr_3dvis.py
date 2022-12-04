import numpy as np
import cv2
import joblib


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
    tcmr_result['frame2index'] = frame2index
    
    # remove redundants
    if not load_verts:
        for person_id in person_ids: 
            tcmr_result[person_id].pop('verts')
        

    return tcmr_result
    
 
def visualize_tcmr_3d(seq_verts, plotting_module = 'pyrender'):

    # 1. load mesh face (We can get from SMPLX) 
    import pickle
    smpl_path = 'smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    #smpl = joblib.load(smpl_path)
    dd = pickle.load(open(smpl_path, 'rb'),  encoding="latin1") # Python3 pickle issue
    print(f"keys: {dd.keys()}")
    faces = dd['f']


    if plotting_module == 'pyrender':
    
        import pyrender
        import trimesh
        
        scene = pyrender.Scene()
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
        vertex_colors = np.ones([seq_verts.shape[1], 4]) * [0.9, 0.5, 0.5, 1.0]
        smplMeshNode = None        
        
        for fn in range(len(seq_verts)):
        
            verts = seq_verts[fn]
            verts =  verts[:, [1,2,0]]
            
            tri_mesh = trimesh.Trimesh(verts, faces, vertex_colors=vertex_colors)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            
            viewer.render_lock.acquire()
            if smplMeshNode is not None:
                scene.remove_node(smplMeshNode)
            smplMeshNode = scene.add(mesh)
            viewer.render_lock.release()
            '''
            if plot_joints:
                sm = trimesh.creation.uv_sphere(radius=0.005)
                sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
                tfs = np.tile(np.eye(4), (len(joints), 1, 1))
                tfs[:, :3, 3] = joints
                joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                scene.add(joints_pcl)
            '''
                                   
        # Close the viwer (and its thread) to finish program     
        viewer.close_external()
        while viewer.is_active:
            pass
               
           
    elif plotting_module == 'matplotlib':
    
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        plt.ion()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        for fn in range(len(seq_verts)):
        
            vertices = seq_verts[fn]
            vertices =  vertices[:, [1,2,0]]
            

            mesh = Poly3DCollection(vertices[faces], alpha=0.1)
            face_color = (0.9, 0.5, 0.5, 1.0)
            edge_color = (0, 0, 0)
            mesh.set_edgecolor(edge_color)
            mesh.set_facecolor(face_color)
            
            ax.add_collection3d(mesh)
            #ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')

            #if plot_joints:
            #    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
            plt.show()
            plt.pause(0.0003)
            ax.collections.pop()
        
    elif plotting_module == 'open3d':
        
        import open3d as o3d


        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='SMPL', width=600, height=600)
        mesh = o3d.geometry.TriangleMesh()
        mesh.triangles = o3d.utility.Vector3iVector(faces)
       
    
        for fn in range(len(seq_verts)):
        
            vertices = seq_verts[fn]
            vertices =  vertices[:, [1,2,0]]
  
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.9, 0.5, 0.5])

            #geometry = [mesh]
            '''
            if plot_joints:
                joints_pcl = o3d.geometry.PointCloud()
                joints_pcl.points = o3d.utility.Vector3dVector(joints)
                joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
                geometry.append(joints_pcl)
            '''
            if fn == 0:
                vis.add_geometry(mesh)
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
            print(f"fn:{fn}")
            #o3d.visualization.draw_geometries(geometry, width = 1000, height = 1000) 
         
        vis.destroy_window()
    
    
if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print(f"usage: python {sys.argv[0]} tcmr_pkl  <fps>")
        exit()

    
    tcmr_result = load_tcmr_result(sys.argv[1], load_verts = True)
    person = tcmr_result[1]
    print(f"keys:{person.keys()}")

    print("Visualizing using Pyrender")
    visualize_tcmr_3d(person['verts'], 'pyrender') 

    print("Visualizing using Open3D")
    visualize_tcmr_3d(person['verts'], 'open3d')

    print("Visualizing using matplotlib")
    visualize_tcmr_3d(person['verts'], 'matplotlib')

