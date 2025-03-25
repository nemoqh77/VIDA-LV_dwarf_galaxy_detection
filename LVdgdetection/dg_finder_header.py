import numpy as np
from astropy.io import fits
import time
import math
import random
from datetime import datetime
import matplotlib.pyplot as plt
#%matplotlib inline
import os
from astropy.io import ascii
import ast
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from multiprocessing import Pool
from scipy.ndimage import convolve

pi=math.pi
cos=math.cos
sin=math.sin
tan=math.tan
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)
def check_import():
    print("1112244")
def load_reg(file):
    with open(file,'r')as f:
        data=f.readlines()
    LBG=[]
    for di in data[3:len(data)]:
        a=di.split(")")[0]
        b=a.split("(")[1].split(",")
        #print(b)
        LBG.append([float(b[0]),float(b[1]),float(b[2])])
    LBG=np.array(LBG)
    return(LBG)
def filter_ms(_data):
    _data0=_data.flatten()
    con0=np.where(abs(_data0-np.mean(_data0))<4*np.std(_data0))
    _data0=_data0[con0[0]]
    con0=np.where(abs(_data0-np.mean(_data0))<4*np.std(_data0))
    _data0=_data0[con0[0]]
    #print(np.mean(patch),np.std(patch))
    con0=np.where(abs(_data0-np.mean(_data0))<4*np.std(_data0))
    _data0=_data0[con0[0]]
    #print(np.mean(patch),np.std(patch))
    std=np.std(_data0)
    mean=np.mean(_data0)
    return(std,mean)

def find_duplicate_columns(arr1, arr2):
    # 转置数组
    arr1_T = arr1.T
    arr2_T = arr2.T
    
    # 对转置后的数组的每一行进行排序
    sorted_arr1 = np.sort(arr1_T, axis=1)
    sorted_arr2 = np.sort(arr2_T, axis=1)
    
    # 使用集合数据结构找出重复的行
    duplicates = set(tuple(row) for row in sorted_arr1).intersection(tuple(row) for row in sorted_arr2)
    
    return len(duplicates)
def divide_sky(data,patch_size,over):
    patch_sky=[]
    l1=int(data.shape[0]/(patch_size-over))-1
    l2=int(data.shape[1]/(patch_size-over))-1
    for i in range(l1):
        x0=i*(patch_size-over)
        for j in range(l2):
            y0=j*(patch_size-over)
            if (y0<data.shape[1])and(x0<data.shape[0]):
                patch_sky.append([x0,int(x0+patch_size),y0,int(y0+patch_size)])
        patch_sky.append([x0,int(x0+patch_size),data.shape[1]-patch_size,data.shape[1]])
    for j in range(l2):
        y0=j*(patch_size-over)
        if (y0<data.shape[1]):
            patch_sky.append([data.shape[0]-patch_size,data.shape[0],y0,int(y0+patch_size)])
    patch_sky.append([data.shape[0]-patch_size,data.shape[0],data.shape[1]-patch_size,data.shape[1]])
    #print("n_subsky:",len(patch_sky))
    return(patch_sky)
def pool_sky(nnp,patch_sky):

    l=int(len(patch_sky)/nnp)
    overn=len(patch_sky)-l*nnp
    #print(overn,l)
    list_p=[]
    for i in range(nnp-overn):
        list_p.append([int(i*l),int(l*(i+1))])
    ssi=int(l*(nnp-overn))
    #print(ssi)
    l=l+1
    for i in range(overn):
        list_p.append([int(ssi+i*l),int(ssi+l*(i+1))])
    return(list_p)

def find_cluster(con1):
    ncluster=1
    #print("pixels over 2_sigma=: ",len(con1[0]))
    cluster_id=np.zeros(len(con1[0]))
    cluster_list=[]
    for i in range(len(con1[0])):
        if cluster_id[i]==0:
            x0=con1[0][i]
            y0=con1[1][i]
            if ncluster==1:
                con=np.where(abs(con1[0]-x0)+abs(con1[1]-y0)<=2)
                if len(con[0])>1:
                    #print("find1",x0,y0,len(con[0])+1)
                    cluster_id[con[0]]=ncluster
                    cluster_id[i]=ncluster
                    for ti in range(150):
                        conn=np.where(cluster_id==ncluster)
                        #print(len(conn[0]))
                        for j in range(len(conn[0])):
                            x1=con1[0][conn[0]][j]
                            y1=con1[1][conn[0]][j]
                            conn1=np.where(abs(con1[0]-x1)+abs(con1[1]-y1)<=2)
                            if len(conn[0])>1:
                                #print("find1",x0,y0,len(conn1[0])+1)
                                cluster_id[conn1[0]]=ncluster
                    conn=np.where(cluster_id==ncluster)
                    #print("cluster1",len(conn[0]))
                else:
                    cluster_id[i]=ncluster
                ncluster+=1
            else:
                #print("try new cluster",ncluster)
                find_cluster=False
                for ci in range(ncluster-1):
                    con=np.where((abs(con1[0]-x0)+abs(con1[1]-y0)<=2)&(cluster_id==ci+1))
                    if len(con[0])>1:
                        #print(x0,y0,ci+1)
                        cluster_id[i]=ci+1
                        find_cluster=True
                if not find_cluster:
                    con=np.where((abs(con1[0]-x0)+abs(con1[1]-y0)<=2)&(cluster_id==0))
                    if len(con[0])>1:
                        cluster_id[con[0]]=ncluster
                        cluster_id[i]=ncluster
                        for ti in range(150):
                            conn=np.where(cluster_id==ncluster)
                            for j in range(len(conn[0])):
                                x1=con1[0][conn[0]][j]
                                y1=con1[1][conn[0]][j]
                                conn1=np.where(abs(con1[0]-x1)+abs(con1[1]-y1)<=2)
                                if len(conn[0])>1:
                                    #print("find1",x0,y0,len(conn1[0])+1)
                                    cluster_id[conn1[0]]=ncluster
                    else:
                        cluster_id[i]=ncluster
                    ncluster+=1
    #print("ncluster: ",ncluster-1)
    return(cluster_id)

def n1find_cluster(con1):
    #print("find cluster start")
    #now = datetime.now()
    #print(f"当前时间：{now}")
    ncluster=1
    #print("pixels over 2_sigma=: ",len(con1[0]))
    cluster_id=np.zeros(len(con1[0]))
    cluster_list=[]
    #print("len(con1[0])",len(con1[0]))
    for i in range(len(con1[0])):
        #print("-------------------------------------------",i)
        if cluster_id[i]==0:
            x0=con1[0][i]
            y0=con1[1][i]
            if ncluster==1:
                con=np.where(abs(con1[0]-x0)+abs(con1[1]-y0)<=2)
                if len(con[0])>1:
                    #print("find1",x0,y0,len(con[0])+1)
                    cluster_id[con[0]]=ncluster
                    cluster_id[i]=ncluster
                    nfinds=[]
                    for ti in range(150):
                        nconn1=np.where(cluster_id==ncluster)
                        if ti==0:
                            conn=np.where(cluster_id==ncluster)
                            l_conn=[]
                        else:
                            conn=[l_conn]
                            l_conn=[]
                        #print(len(conn[0]))
                        nfinds.append(len(nconn1[0]))
                        #print("conn[0]",conn[0])
                        #print("nfinds",nfinds)
                        if len(nfinds)>2:
                            if nfinds[ti]==nfinds[int(ti-1)]:
                                break
                        for j in range(len(conn[0])):
                            x1=con1[0][conn[0]][j]
                            y1=con1[1][conn[0]][j]
                            conn1=np.where((abs(con1[0]-x1)+abs(con1[1]-y1)<=2)&(cluster_id==0))
                            if len(conn1[0])>1:
                                #print("find1",x0,y0,len(conn1[0])+1)
                                cluster_id[conn1[0]]=ncluster
                                l_conn+=list(conn1[0])
                        
                    #conn=np.where(cluster_id==ncluster)
                    #print("cluster1",len(conn[0]))
                else:
                    cluster_id[i]=ncluster
                ncluster+=1
            else:
                #print("try new cluster",ncluster)
                find_cluster=False
                '''
                for ci in range(ncluster-1):
                    con=np.where((abs(con1[0]-x0)+abs(con1[1]-y0)<=2)&(cluster_id==ci+1))
                    if len(con[0])>1:
                        #print(x0,y0,ci+1)
                        cluster_id[i]=ci+1
                        find_cluster=True
                '''
                
                con=np.where((abs(con1[0]-x0)+abs(con1[1]-y0)<=2)&(cluster_id!=0))
                if len(con[0])>1:
                    cluster_id[i]=cluster_id[con[0]][0]
                    find_cluster=True
                if not find_cluster:
                    con=np.where((abs(con1[0]-x0)+abs(con1[1]-y0)<=2)&(cluster_id==0))
                    if len(con[0])>1:
                        cluster_id[con[0]]=ncluster
                        cluster_id[i]=ncluster
                        nfinds=[]
                        for ti in range(150):
                            #print("ti",ti)
                            nconn1=np.where(cluster_id==ncluster)
                            if ti==0:
                                conn=np.where(cluster_id==ncluster)
                                l_conn=[]
                            else:
                                conn=[l_conn]
                                l_conn=[]
                            #print(len(conn[0]))
                            nfinds.append(len(nconn1[0]))
                            if len(nfinds)>2:
                                if nfinds[ti]==nfinds[int(ti-1)]:
                                    break
                            #print("len(conn[0])",len(conn[0]))
                            for j in range(len(conn[0])):
                                x1=con1[0][conn[0]][j]
                                y1=con1[1][conn[0]][j]
                                conn1=np.where((abs(con1[0]-x1)+abs(con1[1]-y1)<=2)&(cluster_id==0))
                                if len(conn1[0])>1:
                                    #print("find1",x0,y0,len(conn1[0])+1)
                                    cluster_id[conn1[0]]=ncluster
                                    l_conn+=list(conn1[0])
                            
                    else:
                        cluster_id[i]=ncluster
                    ncluster+=1
    #print("ncluster: ",ncluster-1)
    #print("find cluster done")
    #now = datetime.now()
    #print(f"当前时间：{now}")
    return(cluster_id)

def _mask_(if0,data,skyi,kernel,kernel1,thre_01,thre_02,thre_03,thre_area1,thre_area2,thre_area3,thre_over1,thre_over2):
    #thre_01  
    #thre_02
    #thre_03
    #thre_area1
    #thre_area2
    #thre_area3,thre_over1,thre_over2
    #candi=[]
    mask=[]
    locals()['datar'+str(if0)]=np.array(data)
    patch=data[skyi[0]:skyi[1],skyi[2]:skyi[3]]
    patch0=np.array(patch)
    copy_patch=np.array(patch)
    map_check=np.zeros((patch.shape))
    output = convolve(patch, kernel)
    npatch=output
    std0,mean0=filter_ms(patch)
    std,mean=filter_ms(npatch)
    con1=np.where((npatch-mean>thre_01*std))
    con2=np.where((npatch-mean>thre_02*std))
    cluster_id=n1find_cluster(con1)
    list_candi1=[]
    list_candi2=[]
    if len(cluster_id)>0:
    
        ncluster=np.max(cluster_id)
        #print("ncluster",ncluster)
        cluster_sn=np.zeros(len(con1[0]))
        c_map_check=np.array(map_check)
        for ci in range(int(ncluster)):
            con=np.where(cluster_id==ci+1)
            for pixel in range(len(con[0])):
                c_map_check[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=(ci+1)
            #c_map_check[:,con1[con[0]]]=ci+1
        for pixel in range(len(con2[0])):
            c_map_check[con2[0][pixel]][con2[1][pixel]]=ncluster+10
        for i in range(len(con1[0])):
            con=np.where((abs(con2[0]-con1[0][i])+abs(con2[1]-con1[1][i]))==0)
            if len(con[0])>0:
                cluster_sn[i]=1
        con1=np.array(con1)
        list_candi1=[]
        for ci in range(int(ncluster)):
            con=np.where((cluster_id==ci+1))
            conn=np.where((cluster_id==ci+1)&(cluster_sn==1))
            #print(ci+1, "  sn>3:"+str(len(con[0])).ljust(10),"       sn>28:",len(conn[0]))
            if len(con[0])>thre_area2 or len(con[0])>thre_area1:
                #if len(conn[0])>0:
                if len(conn[0])/len(con[0])>thre_over1:
                    mask.append([skyi,con1[:,con[0]]])   
                    for pixel in range(len(con[0])):
                        patch0[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=np.random.normal(mean0, std0)
                        map_check[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=(ci+1)
                    if len(con[0])>thre_area2 and len(conn[0])>thre_area1:
                        list_candi1.append([con1[:,con[0]][0]+skyi[0],con1[:,con[0]][1]+skyi[2]])

        #print("////////!!!!!!!!!!!")
        #434 [4200, 4600, 4200, 4600]
        '''
        if (skyi[0]==4200)and(skyi[2]==4200):
        #if 1==1:
            #print("1   ////////!!!!!!!!!!!")
            plt.figure(figsize=(16,5))
            plt.subplot(1,3,1)
            plt.imshow(((copy_patch-mean)/std).transpose(),vmin=-2,vmax=15)
            plt.gca().invert_yaxis()
            plt.title("HBG: band"+str(bi)+"_"+str(if0))
            plt.colorbar()
            plt.subplot(1,3,2)
            #plt.xlim(0,150)
            #plt.ylim(0,150)
            plt.imshow(((patch0-mean)/std).transpose(),vmin=-2,vmax=15)
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.title(if0)
            plt.subplot(1,3,3)
            #plt.xlim(0,150)
            #plt.ylim(0,150)
            plt.imshow(map_check.transpose())
            plt.gca().invert_yaxis()
            plt.title(skyi)
            plt.colorbar()
            plt.show()
        '''

        locals()['datar'+str(if0)][skyi[0]:skyi[1],skyi[2]:skyi[3]]=patch0
        output = convolve(patch0, kernel1)
        patch0=output
        std,mean=filter_ms(patch0)
        map_check=np.zeros((patch0.shape))
        con1=np.where((patch0-mean>thre_03*std))
        cluster_id=n1find_cluster(con1)
        if len(cluster_id)>0:
            ncluster=np.max(cluster_id)
            con1=np.array(con1)

            cci=0
            list_candi2=[]
            for ci in range(int(ncluster)):
                max_over=0
                con=np.where((cluster_id==ci+1))
                if len(con[0])>thre_area3:
                    #print(np.mean(con1[:,con[0]][0]),np.mean(con1[:,con[0]][1]),len(con[0]))
                    cci+=1
                    ccandi2=[con1[:,con[0]][0]+skyi[0],con1[:,con[0]][1]+skyi[2]]
                    for pos_1 in list_candi1:
                        over_pos=find_duplicate_columns(np.array(ccandi2),np.array(pos_1))
                        #print("over:",over_pos,len(pos_1[0]),len(con1[:,con[0]][0]))
                        max_over=max(max_over,over_pos/len(pos_1[0]))
                    if max_over<thre_over2:
                        #nlist_candi2.append(candi_pos)
                        for pixel in range(len(con[0])):
                            map_check[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=cci
                        #list_candi2.append(con1[:,con[0]])
                        list_candi2.append([con1[:,con[0]][0]+skyi[0],con1[:,con[0]][1]+skyi[2]])
                    else:
                        print("\n..max over  "+str(max_over))

        
    else:    
        print("cluster_id is an empty array.")
       
        
    return(list_candi1,list_candi2,locals()['datar'+str(if0)])
'''
def show_mask_(if0,data,skyi,kernel,kernel1,thre_01,thre_02,thre_03,thre_area1,thre_area2,thre_area3,thre_over1,thre_over2):
    #thre_01  
    #thre_02
    #thre_03
    #thre_area1
    #thre_area2
    #thre_area3,thre_over1,thre_over2
    #candi=[]
    mask=[]
    locals()['datar'+str(if0)]=np.array(data)
    patch=data[skyi[0]:skyi[1],skyi[2]:skyi[3]]
    patch0=np.array(patch)
    copy_patch=np.array(patch)
    map_check=np.zeros((patch.shape))
    output = convolve(patch, kernel)
    npatch=output
    std0,mean0=filter_ms(patch)
    std,mean=filter_ms(npatch)
    con1=np.where((npatch-mean>thre_01*std))
    con2=np.where((npatch-mean>thre_02*std))
    cluster_id=n1find_cluster(con1)
    ncluster=np.max(cluster_id)
    #print("ncluster",ncluster)
    cluster_sn=np.zeros(len(con1[0]))
    c_map_check=np.array(map_check)
    for ci in range(int(ncluster)):
        con=np.where(cluster_id==ci+1)
        for pixel in range(len(con[0])):
            c_map_check[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=(ci+1)
        #c_map_check[:,con1[con[0]]]=ci+1
    for pixel in range(len(con2[0])):
        c_map_check[con2[0][pixel]][con2[1][pixel]]=ncluster+10
    for i in range(len(con1[0])):
        con=np.where((abs(con2[0]-con1[0][i])+abs(con2[1]-con1[1][i]))==0)
        if len(con[0])>0:
            cluster_sn[i]=1
    con1=np.array(con1)
    list_candi1=[]
    for ci in range(int(ncluster)):
        con=np.where((cluster_id==ci+1))
        conn=np.where((cluster_id==ci+1)&(cluster_sn==1))
        #print(ci+1, "  sn>3:"+str(len(con[0])).ljust(10),"       sn>28:",len(conn[0]))
        if len(con[0]>thre_area1):
            if len(conn[0])/len(con[0])>thre_over1:
                mask.append([skyi,con1[:,con[0]]])   
                for pixel in range(len(con[0])):
                    patch0[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=np.random.normal(mean0, std0)
                    map_check[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=(ci+1)
                if len(conn[0])>thre_area2:
                    list_candi1.append([con1[:,con[0]][0]+skyi[0],con1[:,con[0]][1]+skyi[2]])
      
    if 1==1:
        #print("1   ////////!!!!!!!!!!!")
        plt.figure(figsize=(16,5))
        plt.subplot(1,3,1)
        plt.imshow(((copy_patch-mean)/std).transpose(),vmin=-2,vmax=15)
        plt.gca().invert_yaxis()
        plt.title("HBG:"+str(if0))
        plt.colorbar()
        plt.subplot(1,3,2)
        #plt.xlim(0,150)
        #plt.ylim(0,150)
        plt.imshow(((patch0-mean)/std).transpose(),vmin=-2,vmax=15)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title(if0)
        plt.subplot(1,3,3)
        #plt.xlim(0,150)
        #plt.ylim(0,150)
        plt.imshow(map_check.transpose())
        plt.gca().invert_yaxis()
        plt.title(skyi)
        plt.colorbar()
        plt.show()
            
    locals()['datar'+str(if0)][skyi[0]:skyi[1],skyi[2]:skyi[3]]=patch0
    output = convolve(patch0, kernel1)
    patch0=output
    std,mean=filter_ms(patch0)
    map_check=np.zeros((patch0.shape))
    con1=np.where((patch0-mean>thre_03*std))
    cluster_id=n1find_cluster(con1)
    ncluster=np.max(cluster_id)
    con1=np.array(con1)
            
    cci=0
    list_candi2=[]
    for ci in range(int(ncluster)):
        max_over=0
        con=np.where((cluster_id==ci+1))
        if len(con[0])>thre_area3:
            #print(np.mean(con1[:,con[0]][0]),np.mean(con1[:,con[0]][1]),len(con[0]))
            cci+=1
            ccandi2=[con1[:,con[0]][0]+skyi[0],con1[:,con[0]][1]+skyi[2]]
            for pos_1 in list_candi1:
                over_pos=find_duplicate_columns(np.array(ccandi2),np.array(pos_1))
                #print("over:",over_pos,len(pos_1[0]),len(con1[:,con[0]][0]))
                max_over=max(max_over,over_pos/len(pos_1[0]))
            if max_over<thre_over2:
                #nlist_candi2.append(candi_pos)
                for pixel in range(len(con[0])):
                    map_check[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=cci
                #list_candi2.append(con1[:,con[0]])
                list_candi2.append([con1[:,con[0]][0]+skyi[0],con1[:,con[0]][1]+skyi[2]])
            else:
                print("\n..max over  "+str(max_over))
    if 1==1:
        plt.figure(figsize=(16,5))
        plt.subplot(1,2,1)
        plt.imshow(((patch0-mean)/std).transpose(),vmin=-1,vmax=5)
        plt.title("LBG:"+str(if0))
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.subplot(1,2,2)
        #plt.xlim(0,150)
        #plt.ylim(0,150)
        plt.imshow(map_check.transpose())
        plt.gca().invert_yaxis()
        plt.title(skyi)
        plt.colorbar()
        plt.show()
    return(list_candi1,list_candi2,locals()['datar'+str(if0)])'''

def show_mask_(if0,data,skyi,kernel,kernel1,thre_01,thre_02,thre_03,thre_area1,thre_area2,thre_area3,thre_over1,thre_over2,x00,y00):
    #thre_01  
    #thre_02
    #thre_03
    #thre_area1
    #thre_area2
    #thre_area3,thre_over1,thre_over2
    #candi=[]
    mask=[]
    locals()['datar'+str(if0)]=np.array(data)
    patch=data[skyi[0]:skyi[1],skyi[2]:skyi[3]]
    patch0=np.array(patch)
    copy_patch=np.array(patch)
    map_check=np.zeros((patch.shape))
    output = convolve(patch, kernel)
    npatch=output
    std0,mean0=filter_ms(patch)
    std,mean=filter_ms(npatch)
    con1=np.where((npatch-mean>thre_01*std))
    con2=np.where((npatch-mean>thre_02*std))
    cluster_id=nfind_cluster(con1)
    #print("HGB cluster:",cluster_id)
    ncluster=np.max(cluster_id)
    #print("ncluster",ncluster)
    cluster_sn=np.zeros(len(con1[0]))
    c_map_check=np.array(map_check)
    for ci in range(int(ncluster)):
        con=np.where(cluster_id==ci+1)
        for pixel in range(len(con[0])):
            c_map_check[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=(ci+1)
        #c_map_check[:,con1[con[0]]]=ci+1
    for pixel in range(len(con2[0])):
        c_map_check[con2[0][pixel]][con2[1][pixel]]=ncluster+10
    for i in range(len(con1[0])):
        con=np.where((abs(con2[0]-con1[0][i])+abs(con2[1]-con1[1][i]))==0)
        if len(con[0])>0:
            cluster_sn[i]=1
    con1=np.array(con1)
    list_candi1=[]
    for ci in range(int(ncluster)):
        con=np.where((cluster_id==ci+1))
        conn=np.where((cluster_id==ci+1)&(cluster_sn==1))
        #print(np.array(con1[con]).shape)
        if (np.min(con1[:,con[0]][0])-x00)*(np.max(con1[:,con[0]][0])-x00)<0 and (np.min(con1[:,con[0]][1])-y00)*(np.max(con1[:,con[0]][1])-y00)<0:
            print("HBG candi dg:",len(con[0]),len(conn[0]))
        #print(ci+1, "  sn>3:"+str(len(con[0])).ljust(10),"       sn>28:",len(conn[0]))
        if len(con[0])>thre_area2 or len(conn[0])>thre_area1:
            if len(conn[0])/len(con[0])>thre_over1:
                mask.append([skyi,con1[:,con[0]]])   
                for pixel in range(len(con[0])):
                    patch0[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=np.random.normal(mean0, std0)
                    map_check[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=(ci+1)
                if len(con[0])>thre_area2 and len(conn[0])>thre_area1:
                    list_candi1.append([con1[:,con[0]][0]+skyi[0],con1[:,con[0]][1]+skyi[2],cluster_sn[con]])
      
    if 1==1:
        print("1   ////////!!!!!!!!!!!")
        plt.figure(figsize=(16,5))
        plt.subplot(1,3,1)
        plt.imshow(((copy_patch-mean)/std).transpose(),vmin=-1,vmax=10)
        plt.gca().invert_yaxis()
        plt.title("S/N Image")
        plt.colorbar()
        plt.subplot(1,3,2)
        #plt.xlim(0,150)
        #plt.ylim(0,150)
        plt.imshow(((patch0-mean)/std).transpose(),vmin=-1,vmax=10)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("masked S/N Image")
        plt.subplot(1,3,3)
        #plt.xlim(0,150)
        #plt.ylim(0,150)
        plt.title("masked objects")
        plt.imshow(map_check.transpose())
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.show()
        plt.close()
        show_sn_map00=((copy_patch-mean0)/std0).transpose()
        show_sn_map0=((npatch-mean)/std).transpose()
        show_sn_map1=((patch0-mean0)/std0).transpose()
        show_sn_map2=map_check.transpose()
        
            
    locals()['datar'+str(if0)][skyi[0]:skyi[1],skyi[2]:skyi[3]]=patch0
    output = convolve(patch0, kernel1)
    patch0=output
    std,mean=filter_ms(patch0)
    map_check=np.zeros((patch0.shape))
    con1=np.where((patch0-mean>thre_03*std))
    cluster_id=nfind_cluster(con1)
    ncluster=np.max(cluster_id)
    con1=np.array(con1)
            
    cci=0
    list_candi2=[]
    for ci in range(int(ncluster)):
        max_over=0
        con=np.where((cluster_id==ci+1))
        if (np.min(con1[:,con[0]][0])-x00)*(np.max(con1[:,con[0]][0])-x00)<0 and (np.min(con1[:,con[0]][1])-y00)*(np.max(con1[:,con[0]][1])-y00)<0:
            print("LBG:candi dg:",len(con[0]),len(conn[0]))
            
        if len(con[0])>thre_area3:
            print(np.mean(con1[:,con[0]][0]),np.mean(con1[:,con[0]][1]),len(con[0]))
            cci+=1
            ccandi2=[con1[:,con[0]][0]+skyi[0],con1[:,con[0]][1]+skyi[2]]
            for pos_1 in list_candi1:
                over_pos=find_duplicate_columns(np.array(ccandi2),np.array(pos_1))
                #print("over:",over_pos,len(pos_1[0]),len(con1[:,con[0]][0]))
                max_over=max(max_over,over_pos/len(pos_1[0]))
            if max_over<thre_over2:
                #nlist_candi2.append(candi_pos)
                for pixel in range(len(con[0])):
                    map_check[con1[0][con[0][pixel]]][con1[1][con[0][pixel]]]=cci
                #list_candi2.append(con1[:,con[0]])
                list_candi2.append([con1[:,con[0]][0]+skyi[0],con1[:,con[0]][1]+skyi[2],np.ones(len(con1[:,con[0]][0]))])
            else:
                print("\n..max over  "+str(max_over))
    if 1==1:
        #print("!!!!!! 222")
        plt.figure(figsize=(16,5))
        plt.subplot(1,2,1)
        plt.imshow(((patch0-mean)/std).transpose(),vmin=-1,vmax=5)
        show_sn_map3=((patch0-mean)/std).transpose()
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.subplot(1,2,2)
        #plt.xlim(0,150)
        #plt.ylim(0,150)
        plt.title("LBG candidates")
        plt.imshow(map_check.transpose())
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.show()
        plt.close()
        
    return(list_candi1,list_candi2,locals()['datar'+str(if0)],show_sn_map00,show_sn_map0,show_sn_map1,show_sn_map2,show_sn_map3,map_check)
def nfind_cluster(con1):
    #print("find cluster start")
    #now = datetime.now()
    #print(f"当前时间：{now}")
    ncluster=1
    #print("pixels over 2_sigma=: ",len(con1[0]))
    cluster_id=np.zeros(len(con1[0]))
    cluster_list=[]
    #print("len(con1[0])",len(con1[0]))
    for i in range(len(con1[0])):
        #print("-------------------------------------------",i)
        if cluster_id[i]==0:
            x0=con1[0][i]
            y0=con1[1][i]
            if ncluster==1:
                con=np.where(abs(con1[0]-x0)+abs(con1[1]-y0)<=2)
                if len(con[0])>1:
                    #print("find1",x0,y0,len(con[0])+1)
                    cluster_id[con[0]]=ncluster
                    cluster_id[i]=ncluster
                    nfinds=[]
                    for ti in range(300):
                        nconn1=np.where(cluster_id==ncluster)
                        if ti==0:
                            conn=np.where(cluster_id==ncluster)
                            l_conn=[]
                        else:
                            conn=[l_conn]
                            l_conn=[]
                        #print(len(conn[0]))
                        nfinds.append(len(nconn1[0]))
                        #print("conn[0]",conn[0])
                        #print("nfinds",nfinds)
                        if len(nfinds)>2:
                            if nfinds[ti]==nfinds[int(ti-1)]:
                                break
                        for j in range(len(conn[0])):
                            x1=con1[0][conn[0]][j]
                            y1=con1[1][conn[0]][j]
                            conn1=np.where((abs(con1[0]-x1)+abs(con1[1]-y1)<=2)&(cluster_id==0))
                            if len(conn1[0])>1:
                                #print("find1",x0,y0,len(conn1[0])+1)
                                cluster_id[conn1[0]]=ncluster
                                l_conn+=list(conn1[0])
                        
                    #conn=np.where(cluster_id==ncluster)
                    #print("cluster1",len(conn[0]))
                else:
                    cluster_id[i]=ncluster
                ncluster+=1
            else:
                #print("try new cluster",ncluster)
                find_cluster=False
                '''
                for ci in range(ncluster-1):
                    con=np.where((abs(con1[0]-x0)+abs(con1[1]-y0)<=2)&(cluster_id==ci+1))
                    if len(con[0])>1:
                        #print(x0,y0,ci+1)
                        cluster_id[i]=ci+1
                        find_cluster=True
                '''
                
                con=np.where((abs(con1[0]-x0)+abs(con1[1]-y0)<=2)&(cluster_id!=0))
                if len(con[0])>1:
                    cluster_id[i]=cluster_id[con[0]][0]
                    find_cluster=True
                if not find_cluster:
                    con=np.where((abs(con1[0]-x0)+abs(con1[1]-y0)<=2)&(cluster_id==0))
                    if len(con[0])>1:
                        cluster_id[con[0]]=ncluster
                        cluster_id[i]=ncluster
                        nfinds=[]
                        for ti in range(300):
                            #print("ti",ti)
                            nconn1=np.where(cluster_id==ncluster)
                            if ti==0:
                                conn=np.where(cluster_id==ncluster)
                                l_conn=[]
                            else:
                                conn=[l_conn]
                                l_conn=[]
                            #print(len(conn[0]))
                            nfinds.append(len(nconn1[0]))
                            if len(nfinds)>2:
                                if nfinds[ti]==nfinds[int(ti-1)]:
                                    break
                            #print("len(conn[0])",len(conn[0]))
                            for j in range(len(conn[0])):
                                x1=con1[0][conn[0]][j]
                                y1=con1[1][conn[0]][j]
                                conn1=np.where((abs(con1[0]-x1)+abs(con1[1]-y1)<=2)&(cluster_id==0))
                                if len(conn1[0])>1:
                                    #print("find1",x0,y0,len(conn1[0])+1)
                                    cluster_id[conn1[0]]=ncluster
                                    l_conn+=list(conn1[0])
                            
                    else:
                        cluster_id[i]=ncluster
                    ncluster+=1
    #print("ncluster: ",ncluster-1)
    #print("find cluster done")
    #now = datetime.now()
    #print(f"当前时间：{now}")
    return(cluster_id)
def nwrite_reg(nnp,path):
    now = datetime.now()
    print(f"start-write_reg--------当前时间：{now}")
    for bi in range(1):
        mkdir(path+"reg/")
        #sky=np.zeros((data.shape))
        with open(path+"reg/LGB_candi.reg",'w')as f:
            f.write("# Region file format: DS9 version 4.1")
            f.write('''\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1''')
            f.write("\nphysical")
        f.close()
        with open(path+"reg/HGB_candi.reg",'w')as f:
            f.write("# Region file format: DS9 version 4.1")
            f.write('''\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1''')
            f.write("\nphysical")
        f.close()

        for nnc in range(2):
            #plt.figure(figsize=(7,7))
            find_candi=0
            alist_candi1=[]
            map_alist_candi1=[]
            for if0 in range(nnp):
                file=path+str(bi)+"_candi_list"+str(if0)+".npy"
                candi=np.load(file,allow_pickle=True)
                #print(if0,candi)
                nci=0
                #print("candi",len(candi))
                for ci in candi: 
                    #print("-----------------------------------")
                    #print(if0)
                    if len(ci[nnc])>0:
                        #print("===========================================================")
                        #print("len candi",if0,nnc,(len(ci[nnc])))
                        #print("===========================================================")
                        for candi_lbg in ci[nnc]:
                            #print(candi_lbg)
                            #candi_lbg = tuple(map(np.array, candi_lbg))
                            scale=2
                            thre_over=0.7
                            #print("-----------------",int(np.mean(candi_lbg[0])),int(np.mean(candi_lbg[1])))
                            ddx,ddy,cen_x,cen_y=find_center(candi_lbg,scale)
                            if len(alist_candi1)==0:
                                alist_candi1.append([cen_y,cen_x,ddx,ddy])
                                map_alist_candi1.append(candi_lbg)
                                find_candi+=1
                            else:
                                nnalist_candi1=np.array(alist_candi1)
                                con=np.where((abs(nnalist_candi1[:,1]-cen_x)<300)&(abs(nnalist_candi1[:,0]-cen_y)<300))
                                if len(con[0])>0:
                                    max_over=0
                                    for pos_1 in map_alist_candi1:
                                        ddx0,ddy0,cen_x0,cen_y0=find_center(pos_1,scale)
                                        if (abs(cen_x0-cen_x)<300)and(abs(cen_x0-cen_x)<300):
                                            if len(np.array(candi_lbg).shape)==1:
                                                copy_candi_lbg=np.vstack((candi_lbg[0],candi_lbg[1]))
                                                candi_lbg=np.array(copy_candi_lbg)
                                            if len(np.array(pos_1).shape)==1:
                                                copy_pos_1=np.vstack((pos_1[0],pos_1[1]))
                                                pos_1=np.array(copy_pos_1)
                                            over_pos=find_duplicate_columns(np.array(candi_lbg),np.array(pos_1))
                                            max_over0=max_over
                                            max_over=max(max_over,over_pos/len(pos_1[0]),over_pos/len(candi_lbg[0]))
                                            if max_over!=max_over0:
                                                #print("new")
                                                _same=pos_1
                                    if max_over>thre_over:
                                        #print("find same",max_over,ddx0,ddy0,cen_x0,cen_y0)
                                        #print("find same",max_over,ddx,ddy,cen_x,cen_y)
                                        #print(_same,candi_lbg)
                                        #print(len(alist_candi1))
                                        #ddx,ddy,cen_x,cen_y=find_center(candi_lbg,scale)
                                        ddx0,ddy0,cen_x0,cen_y0=find_center(_same,scale)
                                        #print("find same",max_over,ddx0,ddy0,cen_x0,cen_y0)
                                        #print("find same",max_over,ddx,ddy,cen_x,cen_y)
                                        copy_a=np.array(alist_candi1)
                                        con=np.where((abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0)==np.min(abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0))))
                                        #print(con[0],np.min(abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0)))
                                        #alist_candi1.remove([cen_y0,cen_x0,ddx0,ddy0])
                                        alist_candi1.remove(alist_candi1[con[0][0]])
                                        #print(len(alist_candi1))
                                        alist_candi1.append([(cen_y+cen_y0)/2,(cen_x+cen_x0)/2,(ddx+ddx0)/2,(ddy0+ddy)/2])
                                        find_candi+=1
                                    else:
                                        '''
                                        if abs(cen_x-1500)+abs(cen_y-1500)<20:
                                            print(cen_x0,cen_y0,cen_y,cen_x,ddx,ddy)'''
                                        alist_candi1.append([cen_y,cen_x,ddx,ddy])
                                        map_alist_candi1.append(candi_lbg)
                                        find_candi+=1
                                else:
                                    alist_candi1.append([cen_y,cen_x,ddx,ddy])
                                    map_alist_candi1.append(candi_lbg)
                                    find_candi+=1
          
            if nnc==1:
                #print("len LBG",find_candi)
                if find_candi>0:
                    for candi in alist_candi1:
                        cen_y,cen_x,ddx,ddy=candi
                        #plt.scatter(cen_x,cen_y)
                        #print(cen_y,cen_x)
                        #print("LBG")
                        with open(path+"reg/LGB_candi.reg",'a+')as f:
                            f.write("\ncircle("+str(cen_y)+","+str(cen_x)+","+str(max(ddx,ddx))+")")
                        f.close()
                        #plt.scatter(cen_y,cen_x,c='r')
            else:
                #print("len HBG",find_candi)
                if find_candi>0:
                    for candi in alist_candi1:
                        cen_y,cen_x,ddx,ddy=candi
                        #plt.scatter(cen_x,cen_y)
                        #print(cen_y,cen_x)
                        #print("HBG")
                        with open(path+"reg/HGB_candi.reg",'a+')as f:
                            f.write("\ncircle("+str(cen_y)+","+str(cen_x)+","+str(max(ddx,ddx))+")")
                        f.close()
                        #plt.scatter(cen_y,cen_x,c='b')
    now = datetime.now()
    #print(f"done-write_reg--------当前时间：{now}")

def n1write_reg(orig_data,nnp,path,bi):
    now = datetime.now()
    print(f"start-write_reg--------当前时间：{now}")
    bi_scamp_candi=[]
    for bbi in range(1):
        mkdir(path+"reg/")
        #sky=np.zeros((data.shape))
        with open(path+"reg/LGB_candi"+str(bi)+".reg",'w')as f:
            f.write("# Region file format: DS9 version 4.1")
            f.write('''\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1''')
            f.write("\nphysical")
        f.close()
        with open(path+"reg/HGB_candi"+str(bi)+".reg",'w')as f:
            f.write("# Region file format: DS9 version 4.1")
            f.write('''\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1''')
            f.write("\nphysical")
        f.close()
        
        for nnc in range(2):
            #plt.figure(figsize=(7,7))
            find_candi=0
            alist_candi1=[]
            map_alist_candi1=[]
            scamp_candi=[]
            for if0 in range(nnp):
                file=path+str(bi)+"_candi_list"+str(if0)+".npy"
                candi=np.load(file,allow_pickle=True)
                #print(if0,candi)
                nci=0
                #print("candi",len(candi))
                for ci in candi: 
                    #print("-----------------------------------")
                    #print(if0)
                    if len(ci[nnc])>0:
                        #print("===========================================================")
                        #print("len candi",if0,nnc,(len(ci[nnc])))
                        #print("===========================================================")
                        for candi_lbg in ci[nnc]:
                            #print(candi_lbg)
                            #candi_lbg = tuple(map(np.array, candi_lbg))
                            scale=2
                            thre_over=0.7
                            #print("-----------------",int(np.mean(candi_lbg[0])),int(np.mean(candi_lbg[1])))
                            ddx,ddy,cen_x,cen_y=find_center(candi_lbg,scale)
                            if len(alist_candi1)==0:
                                alist_candi1.append([cen_y,cen_x,ddx,ddy])
                                map_alist_candi1.append([candi_lbg,ddx,ddy,cen_x,cen_y])
                                scamp_scale=max(ddx,ddy)*2
                                #print("\\",int(cen_y-scamp_scale),int(cen_y-scamp_scale),int(cen_x-scamp_scale),int(cen_x-scamp_scale))
                                scamp_candi.append(orig_data[max(0,int(cen_y-scamp_scale)):min(len(orig_data),int(cen_y+scamp_scale)),max(0,int(cen_x-scamp_scale)):min(len(orig_data[0]),int(cen_x+scamp_scale))])
                                find_candi+=1
                            else:
                                nnalist_candi1=np.array(alist_candi1)
                                con=np.where((abs(nnalist_candi1[:,1]-cen_x)<300)&(abs(nnalist_candi1[:,0]-cen_y)<300))
                                if len(con[0])>0:
                                    max_over=0
                                    for pos_1a in map_alist_candi1:
                                        pos_1,ddx0,ddy0,cen_x0,cen_y0=pos_1a  #find_center(pos_1,scale)
                                        if (abs(cen_x0-cen_x)<200)and(abs(cen_x0-cen_x)<200):
                                            if len(np.array(candi_lbg).shape)==1:
                                                copy_candi_lbg=np.vstack((candi_lbg[0],candi_lbg[1]))
                                                candi_lbg=np.array(copy_candi_lbg)
                                            if len(np.array(pos_1).shape)==1:
                                                copy_pos_1=np.vstack((pos_1[0],pos_1[1]))
                                                pos_1=np.array(copy_pos_1)
                                            over_pos=find_duplicate_columns(np.array(candi_lbg),np.array(pos_1))
                                            max_over0=max_over
                                            max_over=max(max_over,over_pos/len(pos_1[0]),over_pos/len(candi_lbg[0]))
                                            if max_over!=max_over0:
                                                #print("new")
                                                _same=pos_1
                                                addx0,addy0,acen_x0,acen_y0=ddx0,ddy0,cen_x0,cen_y0
                                    if max_over>thre_over:
                                        #print("find same",max_over,ddx0,ddy0,cen_x0,cen_y0)
                                        #print("find same",max_over,ddx,ddy,cen_x,cen_y)
                                        #print(_same,candi_lbg)
                                        #print(len(alist_candi1))
                                        #ddx,ddy,cen_x,cen_y=find_center(candi_lbg,scale)
                                        addx0,addy0,acen_x0,acen_y0=ddx0,ddy0,cen_x0,cen_y0 #=find_center(_same,scale)
                                        #print("find same",max_over,ddx0,ddy0,cen_x0,cen_y0)
                                        #print("find same",max_over,ddx,ddy,cen_x,cen_y)
                                        copy_a=np.array(alist_candi1)
                                        con=np.where((abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0)==np.min(abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0))))
                                        #print(con[0],np.min(abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0)))
                                        #alist_candi1.remove([cen_y0,cen_x0,ddx0,ddy0])
                                        alist_candi1.remove(alist_candi1[con[0][0]])
                                        #scamp_candi.remove(scamp_candi[con[0][0]])
                                        del scamp_candi[con[0][0]]
                                        #print(len(alist_candi1))
                                        alist_candi1.append([(cen_y+cen_y0)/2,(cen_x+cen_x0)/2,(ddx+ddx0)/2,(ddy0+ddy)/2])
                                        scamp_scale=max((ddx+ddx0)/2,(ddy0+ddy)/2)*2
                                        scamp_candi.append(orig_data[max(0,int((cen_y+cen_y0)/2-scamp_scale)):min(len(orig_data),int((cen_y+cen_y0)/2+scamp_scale)),max(0,int((cen_x+cen_x0)/2-scamp_scale)):min(len(orig_data[0]),int((cen_x+cen_x0)/2+scamp_scale))])
                                        find_candi+=1
                                    else:
                                        '''
                                        if abs(cen_x-1500)+abs(cen_y-1500)<20:
                                            print(cen_x0,cen_y0,cen_y,cen_x,ddx,ddy)'''
                                        alist_candi1.append([cen_y,cen_x,ddx,ddy])
                                        map_alist_candi1.append([candi_lbg,ddx,ddy,cen_x,cen_y])
                                        scamp_scale=max(ddx,ddy)*2
                                        scamp_candi.append(orig_data[max(0,int(cen_y-scamp_scale)):min(len(orig_data),int(cen_y+scamp_scale)),max(0,int(cen_x-scamp_scale)):min(len(orig_data[0]),int(cen_x+scamp_scale))])
                                        find_candi+=1
                                else:
                                    alist_candi1.append([cen_y,cen_x,ddx,ddy])
                                    map_alist_candi1.append([candi_lbg,ddx,ddy,cen_x,cen_y])
                                    scamp_scale=max(ddx,ddy)*2
                                    scamp_candi.append(orig_data[max(0,int(cen_y-scamp_scale)):min(len(orig_data),int(cen_y+scamp_scale)),max(0,int(cen_x-scamp_scale)):min(len(orig_data[0]),int(cen_x+scamp_scale))])
                                    find_candi+=1
            print("fcandi",len(alist_candi1))
            if nnc==1:
                #print("len LBG",find_candi)
                if find_candi>0:
                    for candi in alist_candi1:
                        cen_y,cen_x,ddx,ddy=candi
                        #plt.scatter(cen_x,cen_y)
                        #print(cen_y,cen_x)
                        #print("LBG")
                        with open(path+"reg/LGB_candi"+str(bi)+".reg",'a+')as f:
                            f.write("\ncircle("+str(cen_y)+","+str(cen_x)+","+str(max(ddx,ddx))+")")
                        f.close()
                        #plt.scatter(cen_y,cen_x,c='r')
            else:
                #print("len HBG",find_candi)
                if find_candi>0:
                    for candi in alist_candi1:
                        cen_y,cen_x,ddx,ddy=candi
                        #plt.scatter(cen_x,cen_y)
                        #print(cen_y,cen_x)
                        #print("HBG")
                        with open(path+"reg/HGB_candi"+str(bi)+".reg",'a+')as f:
                            f.write("\ncircle("+str(cen_y)+","+str(cen_x)+","+str(max(ddx,ddx))+")")
                        f.close()
                        #plt.scatter(cen_y,cen_x,c='b')
            bi_scamp_candi.append(scamp_candi)
    now = datetime.now()
    #print(f"done-write_reg--------当前时间：{now}")
    return(bi_scamp_candi)
def n1writedg_reg(list_candi):
    find_candi=0
    alist_candi1=[]
    map_alist_candi1=[]
    scamp_candi=[]

            
    for candi_lbg in list_candi:
        #print(candi_lbg)
        #candi_lbg = tuple(map(np.array, candi_lbg))
        scale=2
        thre_over=0.7
        #print("-----------------",int(np.mean(candi_lbg[0])),int(np.mean(candi_lbg[1])))
        ddx,ddy,cen_x,cen_y=find_center(candi_lbg,scale)
        if len(alist_candi1)==0:
            alist_candi1.append([cen_y,cen_x,ddx,ddy])
            find_candi+=1
        else:
            nnalist_candi1=np.array(alist_candi1)
            con=np.where((abs(nnalist_candi1[:,1]-cen_x)<300)&(abs(nnalist_candi1[:,0]-cen_y)<300))
            if len(con[0])>0:
                max_over=0
                for pos_1 in map_alist_candi1:
                    ddx0,ddy0,cen_x0,cen_y0=find_center(pos_1,scale)
                    if (abs(cen_x0-cen_x)<300)and(abs(cen_x0-cen_x)<300):
                        if len(np.array(candi_lbg).shape)==1:
                            copy_candi_lbg=np.vstack((candi_lbg[0],candi_lbg[1]))
                            candi_lbg=np.array(copy_candi_lbg)
                        if len(np.array(pos_1).shape)==1:
                            copy_pos_1=np.vstack((pos_1[0],pos_1[1]))
                            pos_1=np.array(copy_pos_1)
                        over_pos=find_duplicate_columns(np.array(candi_lbg),np.array(pos_1))
                        max_over0=max_over
                        max_over=max(max_over,over_pos/len(pos_1[0]),over_pos/len(candi_lbg[0]))
                        if max_over!=max_over0:
                            #print("new")
                            _same=pos_1
                if max_over>thre_over:
                    #print("find same",max_over,ddx0,ddy0,cen_x0,cen_y0)
                    #print("find same",max_over,ddx,ddy,cen_x,cen_y)
                    #print(_same,candi_lbg)
                    #print(len(alist_candi1))
                    #ddx,ddy,cen_x,cen_y=find_center(candi_lbg,scale)
                    ddx0,ddy0,cen_x0,cen_y0=find_center(_same,scale)
                    #print("find same",max_over,ddx0,ddy0,cen_x0,cen_y0)
                    #print("find same",max_over,ddx,ddy,cen_x,cen_y)
                    copy_a=np.array(alist_candi1)
                    con=np.where((abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0)==np.min(abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0))))
                    #print(con[0],np.min(abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0)))
                    #alist_candi1.remove([cen_y0,cen_x0,ddx0,ddy0])
                    alist_candi1.remove(alist_candi1[con[0][0]])
                    
                    alist_candi1.append([(cen_y+cen_y0)/2,(cen_x+cen_x0)/2,(ddx+ddx0)/2,(ddy0+ddy)/2])
                    find_candi+=1
                else:
                    '''
                    if abs(cen_x-1500)+abs(cen_y-1500)<20:
                        print(cen_x0,cen_y0,cen_y,cen_x,ddx,ddy)'''
                    alist_candi1.append([cen_y,cen_x,ddx,ddy])
                    find_candi+=1
            else:
                alist_candi1.append([cen_y,cen_x,ddx,ddy])
                find_candi+=1

    result=[]
    if find_candi>0:
        for candi in alist_candi1:
            cen_y,cen_x,ddx,ddy=candi
            result.append([cen_y,cen_x,max(ddx,ddx)])

    now = datetime.now()
    #print(f"done-write_reg--------当前时间：{now}")
    return(np.array(result))
def bkgwrite_candi(orig_data,crosscandi,namereg,path,band):
    now = datetime.now()
    #print(f"start-write_reg--------当前时间：{now}")
    mkdir(path+"reg/")
    #print("writereg:",path+"reg/"+namereg+"_candi"+str(band[0])+".reg")
    with open(path+"reg/"+namereg+"_candi"+str(band[0])+".reg",'w')as f:
        f.write("# Region file format: DS9 version 4.1")
        f.write('''\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1''')
        f.write("\nphysical")
    f.close()
    bi=band[0]    
    find_candi=0
    alist_candi1=[]
    scamp_candi=[]
    for candi in crosscandi:       
        scale=2
        cen_y,cen_x=candi[0],candi[1]
        scamp_scale=candi[2]*scale
        alist_candi1.append([cen_y,cen_x,candi[2]])
        scamp_candi.append(orig_data[:,max(0,int(cen_y)-scamp_scale):min(len(orig_data),int(cen_y)+scamp_scale)][max(0,int(cen_x)-scamp_scale):min(len(orig_data[0]),int(cen_x)+scamp_scale)])
        find_candi+=1

    if find_candi>0:
        for candi in alist_candi1:
            cen_y,cen_x,ddx=candi
            with open(path+"reg/"+namereg+"_candi"+str(bi)+".reg",'a+')as f:
                f.write("\ncircle("+str(cen_y)+","+str(cen_x)+","+str(ddx)+")")
            f.close()
    now = datetime.now()
    #print(f"done-write_reg--------当前时间：{now}")
    return(scamp_candi)
def nwrite_reg_hostg(orig_data,nnp,path,dgi,bi):
    now = datetime.now()
    print(f"start-write_reg--------当前时间：{now}")
    bi_scamp_candi=[]
    for bbi in range(1):
        mkdir(path+"reg/")
        #sky=np.zeros((data.shape))
        with open(path+"reg/LGB_candi"+str(dgi)+"_"+str(bi)+".reg",'w')as f:
            f.write("# Region file format: DS9 version 4.1")
            f.write('''\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1''')
            f.write("\nphysical")
        f.close()
        with open(path+"reg/HGB_candi"+str(dgi)+"_"+str(bi)+".reg",'w')as f:
            f.write("# Region file format: DS9 version 4.1")
            f.write('''\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1''')
            f.write("\nphysical")
        f.close()
        
        for nnc in range(2):
            #plt.figure(figsize=(7,7))
            find_candi=0
            alist_candi1=[]
            map_alist_candi1=[]
            scamp_candi=[]
            for if0 in range(nnp):
                file=path+str(dgi)+"_"+str(bi)+"_candi_list"+str(if0)+".npy"
                candi=np.load(file,allow_pickle=True)
                #print(if0,candi)
                nci=0
                #print("candi",len(candi))
                for ci in candi: 
                    #print("-----------------------------------")
                    #print(if0)
                    if len(ci[nnc])>0:
                        #print("===========================================================")
                        #print("len candi",if0,nnc,(len(ci[nnc])))
                        #print("===========================================================")
                        for candi_lbg in ci[nnc]:
                            #print(candi_lbg)
                            #candi_lbg = tuple(map(np.array, candi_lbg))
                            scale=2
                            thre_over=0.7
                            #print("-----------------",int(np.mean(candi_lbg[0])),int(np.mean(candi_lbg[1])))
                            ddx,ddy,cen_x,cen_y=find_center(candi_lbg,scale)
                            if len(alist_candi1)==0:
                                alist_candi1.append([cen_y,cen_x,ddx,ddy])
                                map_alist_candi1.append(candi_lbg)
                                scamp_scale=max(ddx,ddy)*2
                                #print("\\",int(cen_y-scamp_scale),int(cen_y-scamp_scale),int(cen_x-scamp_scale),int(cen_x-scamp_scale))
                                scamp_candi.append(orig_data[max(0,int(cen_y-scamp_scale)):min(len(orig_data),int(cen_y+scamp_scale)),max(0,int(cen_x-scamp_scale)):min(len(orig_data[0]),int(cen_x+scamp_scale))])
                                find_candi+=1
                            else:
                                nnalist_candi1=np.array(alist_candi1)
                                con=np.where((abs(nnalist_candi1[:,1]-cen_x)<300)&(abs(nnalist_candi1[:,0]-cen_y)<300))
                                if len(con[0])>0:
                                    max_over=0
                                    for pos_1 in map_alist_candi1:
                                        ddx0,ddy0,cen_x0,cen_y0=find_center(pos_1,scale)
                                        if (abs(cen_x0-cen_x)<300)and(abs(cen_x0-cen_x)<300):
                                            if len(np.array(candi_lbg).shape)==1:
                                                copy_candi_lbg=np.vstack((candi_lbg[0],candi_lbg[1]))
                                                candi_lbg=np.array(copy_candi_lbg)
                                            if len(np.array(pos_1).shape)==1:
                                                copy_pos_1=np.vstack((pos_1[0],pos_1[1]))
                                                pos_1=np.array(copy_pos_1)
                                            over_pos=find_duplicate_columns(np.array(candi_lbg),np.array(pos_1))
                                            max_over0=max_over
                                            max_over=max(max_over,over_pos/len(pos_1[0]),over_pos/len(candi_lbg[0]))
                                            if max_over!=max_over0:
                                                #print("new")
                                                _same=pos_1
                                    if max_over>thre_over:
                                        #print("find same",max_over,ddx0,ddy0,cen_x0,cen_y0)
                                        #print("find same",max_over,ddx,ddy,cen_x,cen_y)
                                        #print(_same,candi_lbg)
                                        #print(len(alist_candi1))
                                        #ddx,ddy,cen_x,cen_y=find_center(candi_lbg,scale)
                                        ddx0,ddy0,cen_x0,cen_y0=find_center(_same,scale)
                                        #print("find same",max_over,ddx0,ddy0,cen_x0,cen_y0)
                                        #print("find same",max_over,ddx,ddy,cen_x,cen_y)
                                        copy_a=np.array(alist_candi1)
                                        con=np.where((abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0)==np.min(abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0))))
                                        #print(con[0],np.min(abs(copy_a[:,0]-cen_y0)+abs(copy_a[:,1]-cen_x0)))
                                        #alist_candi1.remove([cen_y0,cen_x0,ddx0,ddy0])
                                        alist_candi1.remove(alist_candi1[con[0][0]])
                                        #scamp_candi.remove(scamp_candi[con[0][0]])
                                        del scamp_candi[con[0][0]]
                                        #print(len(alist_candi1))
                                        alist_candi1.append([(cen_y+cen_y0)/2,(cen_x+cen_x0)/2,(ddx+ddx0)/2,(ddy0+ddy)/2])
                                        scamp_scale=max((ddx+ddx0)/2,(ddy0+ddy)/2)*2
                                        scamp_candi.append(orig_data[max(0,int(((cen_y+cen_y0)/2)-scamp_scale)):min(len(orig_data),int(((cen_y+cen_y0)/2)+scamp_scale)),max(0,int((cen_x+cen_x0)/2-scamp_scale)):min(len(orig_data[0]),int((cen_x+cen_x0)/2+scamp_scale))])
                                        find_candi+=1
                                    else:
                                        '''
                                        if abs(cen_x-1500)+abs(cen_y-1500)<20:
                                            print(cen_x0,cen_y0,cen_y,cen_x,ddx,ddy)'''
                                        alist_candi1.append([cen_y,cen_x,ddx,ddy])
                                        map_alist_candi1.append(candi_lbg)
                                        scamp_scale=max(ddx,ddy)*2
                                        scamp_candi.append(orig_data[max(0,int(cen_y-scamp_scale)):min(len(orig_data),int(cen_y+scamp_scale)),max(0,int(cen_x-scamp_scale)):min(len(orig_data[0]),int(cen_x+scamp_scale))])
                                        find_candi+=1
                                else:
                                    alist_candi1.append([cen_y,cen_x,ddx,ddy])
                                    map_alist_candi1.append(candi_lbg)
                                    scamp_scale=max(ddx,ddy)*2
                                    scamp_candi.append(orig_data[max(0,int(cen_y-scamp_scale)):min(len(orig_data),int(cen_y+scamp_scale)),max(0,int(cen_x-scamp_scale)):min(len(orig_data[0]),int(cen_x+scamp_scale))])
                                    find_candi+=1
          
            if nnc==1:
                #print("len LBG",find_candi)
                if find_candi>0:
                    for candi in alist_candi1:
                        cen_y,cen_x,ddx,ddy=candi
                        #plt.scatter(cen_x,cen_y)
                        #print(cen_y,cen_x)
                        #print("LBG")
                        with open(path+"reg/LGB_candi"+str(dgi)+"_"+str(bi)+".reg",'a+')as f:
                            f.write("\ncircle("+str(cen_y)+","+str(cen_x)+","+str(max(ddx,ddx))+")")
                        f.close()
                        #plt.scatter(cen_y,cen_x,c='r')
            else:
                #print("len HBG",find_candi)
                if find_candi>0:
                    for candi in alist_candi1:
                        cen_y,cen_x,ddx,ddy=candi
                        #plt.scatter(cen_x,cen_y)
                        #print(cen_y,cen_x)
                        #print("HBG")
                        with open(path+"reg/HGB_candi"+str(dgi)+"_"+str(bi)+".reg",'a+')as f:
                            f.write("\ncircle("+str(cen_y)+","+str(cen_x)+","+str(max(ddx,ddx))+")")
                        f.close()
                        #plt.scatter(cen_y,cen_x,c='b')
            bi_scamp_candi.append(scamp_candi)
    now = datetime.now()
    #print(f"done-write_reg--------当前时间：{now}")
    return(bi_scamp_candi)
    
def find_center(data_obj,scale):
    ddx=(max(data_obj[0])-min(data_obj[0]))*scale/2
    ddy=(max(data_obj[1])-min(data_obj[1]))*scale/2
    cen_x=np.mean(data_obj[0])
    cen_y=np.mean(data_obj[1])
    return(ddx,ddy,cen_x,cen_y)
