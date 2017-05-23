from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import csv
import os
from glob import glob
import pandas as pd
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

#Some helper functions

def make_mask(center, diam, z, width, height, spacing, origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameterim
width, height : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height, width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing                #鎶婅妭鐐逛腑蹇冭浆鍖栦负浣撶礌鍧愭爣(x,y,z)
    v_diam = int(diam/spacing[0]+5)
    """鎵惧埌x , y鐨勮竟鐣岃寖鍥村苟杩涜�閫傚綋鎵╁�(5)"""
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z])) <= diam:    #np.linalg.norm(x锛�ord=)姹倄鐨勮寖鏁伴粯璁�rd=None涓�鑼冩暟
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0    #杞�洖浣撶礌鍧愭爣锛岀劧鍚庢爣1
    return(mask)

def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))

############
#
# Getting list of image files
luna_path = "./test_subset01/"
luna_subset_path = luna_path+"subset1/"
output_path = "/home/jonathan/tutorial/"
file_list=glob(luna_subset_path+"*.mhd")      #鑾峰彇鎵�湁mhd鏂囦欢鐨勫畬鏁磋矾寰�

#####################
#
# Helper function to get rows in data frame associated 
# with each file
def get_filename(file_list, case):            #鐢ㄤ簬鏌ユ壘case鎵��搴旂殑mhd鏂囦欢
    for f in file_list:
        if case in f:
            return(f)
#
# The locations of the nodes
df_node = pd.read_csv(luna_path+"annotations.csv")    #璇诲彇csv鏂囦欢涓�殑鑺傜偣鏁版嵁
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name)) #鍦╠f_node涓��鍔爁ile椤癸紝瀛樻斁鑺傜偣瀵瑰簲鏂囦欢鐩�綍
df_node = df_node.dropna()                    #鏍规嵁鏍囩�鍊兼槸鍚︾己鎹熷�鍏惰繘琛岃繃婊�

#
# Looping over the image files
#
for fcount, img_file in enumerate(tqdm(file_list)):
    mini_df = df_node[df_node["file"] == img_file] #get all nodules associate with file   鍙栧嚭杩欎釜mhd鏂囦欢瀵瑰簲鐨勭敱CSV鏋勫缓鐨刣ataframe鐨勫�搴旇�
    if mini_df.shape[0]>0:                       # 璺宠繃鍦╟sv涓�病鏈夋爣璁扮殑mhd鏂囦欢--.shape浼氳繑鍥炰竴涓�淮搴︾殑鍏冪粍濡傛湰渚嬫湁鏍囪�鍒�1,6)鏃犳爣璁板垯(0,6)
        # load the data once
        itk_img = sitk.ReadImage(img_file)          #璇诲彇mhd鏂囦欢
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)灏唌hd鏂囦欢杞�寲涓轰笁缁存暟缁�娉ㄦ剰杞寸殑椤哄簭鏄痾,y,z
        num_z, height, width = img_array.shape        #height X width constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm) 鑾峰彇涓栫晫鍧愭爣鐨勫師鐐瑰苟杞�寲鎴恘umpy鏁扮粍
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)   鑾峰彇濉�厖闂磋窛 x,y,z
        # go through all nodes (why just the biggest?)
        for node_idx, cur_row in mini_df.iterrows():  #dataframe.iterrows() 浠ヨ�鐨勫舰寮忚繘琛岃凯浠ｏ紝杩斿洖涓�釜鍏冪粍(index, 琛宒ataframe)
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]               #鍙栬妭鐐逛綅缃�潗鏍�            diam = cur_row["diameter_mm"]           #鍙栬妭鐐瑰崐寰�            # just keep 3 slices
            imgs = np.ndarray([3,height,width],dtype=np.float32)  #鍦╩hd鏂囦欢涓�彇3涓�垏鐗�            masks = np.ndarray([3,height,width],dtype=np.uint8)
            center = np.array([node_x, node_y, node_z])   # 鑺傜偣涓�績(涓栫晫鍧愭爣mm)
            v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)鎶婁笘鐣屽潗鏍囩殑鑺傜偣涓�績杞�崲鎴愪綋绱犲潗鏍�            for i, i_z in enumerate(np.arange(int(v_center[2])-1,  int(v_center[2])+2).clip(0, num_z-1)): # 鍙�涓�垏鐗囧苟涓旈槻姝�鍚戣秺鐣宑lip prevents going out of bounds in Z
            mask = make_mask(center, diam, i_z*spacing[2]+origin[2], width, height, spacing, origin)
            masks[i] = mask                      #i涓哄簭鍙凤紝0绗�竴寮狅紝1绗�簩寮狅紝2绗�笁寮�鍒囩墖
            imgs[i] = img_array[i_z]
            np.save(os.path.join(output_path,"images_%04d_%04d.npy" % (fcount, node_idx)),imgs)
            np.save(os.path.join(output_path,"masks_%04d_%04d.npy" % (fcount, node_idx)),masks)
