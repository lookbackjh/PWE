from cProfile import label
from email.headerregistry import Group
import numpy as np
from sklearn.utils import resample
from src.Grouping import Groupinfo
import matplotlib.pyplot as plt
from scipy.special import rel_entr
import math
import copy
import pandas as pd
import os
import time
from tqdm import tqdm
class Permutator():

    def __init__(self,g1,g2,feat_info,args) -> None:
        self.g1=g1
        self.g2=g2
        self.args=args
        self.feat_info=feat_info
        self.p_feature_posteriors_g1=[]
        self.p_feature_evaluations_g1=[]
        self.p_feature_posteriors_g2=[]
        self.p_feature_evaluations_g2=[]
        pass

    def permutation(self):
        epsilon=self.args.epsilon
        x_grid=np.arange(self.args.x_grid_start,self.args.x_grid_end,0.1)
        #feat_info=np.zeros(self.args.num_feature)
        n,d=np.shape(self.g1)
        z=np.concatenate([self.g1,self.g2],axis=1) # for permutation.
        t1=Groupinfo(self.g1)
        t2=Groupinfo(self.g2)
        t1.finddistribution(x_grid,epsilon,self.feat_info)
        t2.finddistribution(x_grid,epsilon,self.feat_info)
        t1pos,t1eva=t1.summation(x_grid)
        t2pos,t2eva=t2.summation(x_grid)
        k=0
        diff=self.ShannonJanson(t1pos,t2pos,t1eva,t2eva,x_grid)
        print("Distance of Original Data :{}".format(diff))
        plt.legend()
        for i in tqdm(range(self.args.p_num)):
            z=np.take(z,np.random.permutation(z.shape[1]),axis=1,out=z)
            g1=z[:,d:]
            g2=z[:,:d]
            #print(g1.shape)
            t1=Groupinfo(g1)
            t2=Groupinfo(g2)
            t1.finddistribution(x_grid,epsilon,self.feat_info)
            t2.finddistribution(x_grid,epsilon,self.feat_info)
            t1pos,t1eva=t1.summation(x_grid)
            t2pos,t2eva=t2.summation(x_grid)
        
            dis=self.ShannonJanson(t1pos,t2pos,t1eva,t2eva,x_grid)
            print("Distance for iteration{} {}".format(i+1,dis))
            k+=diff<dis
        return (k/self.args.p_num)
    
    def metapermutation_feature(self):
        epsilon=self.args.epsilon
        x_grid=np.arange(self.args.x_grid_start,self.args.x_grid_end,self.args.interval)
        feat_info=np.zeros(self.args.num_feature)
        log=self.args.predefined
        # bwfilename="n{}bw_k{}.txt".format(self.args.num_feature,self.args.same_ratio)
        # sigfilename="n{}sigma_k{}.txt".format(self.args.num_feature,self.args.same_ratio)
        # curdir=(os.getcwd())
        # sig=os.path.join(curdir, 'Generated_Data','sensitivity_by_n',sigfilename)
        # b=os.path.join(curdir,'Generated_Data','sensitivity_by_n',bwfilename)
        
        if self.args.num_samples==20:
            df=pd.read_csv("Generated_Data/sensitivity_by_n/n20_df.csv")
        elif self.args.num_samples==50:
            df=pd.read_csv("Generated_Data/sensitivity_by_n/n50_df.csv")
        
        else:
            df=pd.read_csv("Generated_Data/sensitivity_by_n/n100_df.csv")

        #df=pd.read_csv("Generated_Data/sensitivity_by_n/n50_df.csv")
        sigma=df['sigma']
        bw=df['bandwidth']
        feat_info+=1.5 ## make prior not existing. in simulated case. 

        n,d=np.shape(self.g1)
        z=np.concatenate([self.g1,self.g2],axis=1) # for permutation.
        #for i in tqdm(range(self.args.p_num)):
            #z=np.take(z,np.random.permutation(z.shape[1]),axis=1,out=z)
        g1=z[:,d:]
        g2=z[:,:d]
        #print(g1.shape)
        t1=Groupinfo(g1)
        t2=Groupinfo(g2)
        t1.metadistribution(x_grid,epsilon,bw,sigma,feat_info,log)
        t2.metadistribution(x_grid,epsilon,bw,sigma,feat_info,log)
        t1pos,t1kde,t1bw=t1.get_feature_distribution()
        t2pos,t2kde,t2bw=t2.get_feature_distribution()
        self.p_feature_posteriors_g1.append(t1pos)
        self.p_feature_evaluations_g1.append(t1kde)
        self.p_feature_posteriors_g2.append(t2pos)
        self.p_feature_evaluations_g2.append(t2kde)

        return t1pos,t1kde,t2pos,t2kde,t1bw,t2bw




    def metapermutation(self):
        epsilon=self.args.epsilon
        x_grid=np.arange(self.args.x_grid_start,self.args.x_grid_end,self.args.interval)
        feat_info=np.zeros(self.args.num_feature)
        log=self.args.predefined
        # bwfilename="n{}bw_k{}.txt".format(self.args.num_feature,self.args.same_ratio)
        # sigfilename="n{}sigma_k{}.txt".format(self.args.num_feature,self.args.same_ratio)
        # curdir=(os.getcwd())
        # sig=os.path.join(curdir, 'Generated_Data','sensitivity_by_n',sigfilename)
        # b=os.path.join(curdir,'Generated_Data','sensitivity_by_n',bwfilename)
        
        if self.args.num_samples==20:
            df=pd.read_csv("Generated_Data/sensitivity_by_n/n20_df.csv")
        elif self.args.num_samples==50:
            df=pd.read_csv("Generated_Data/sensitivity_by_n/n50_df.csv")
        
        else:
            df=pd.read_csv("Generated_Data/sensitivity_by_n/n100_df.csv")

        #df=pd.read_csv("Generated_Data/sensitivity_by_n/n50_df.csv")
        sigma=df['sigma']
        bw=df['bandwidth']
        feat_info+=1.5 ## make prior not existing. in simulated case. 

        n,d=np.shape(self.g1)
        z=np.concatenate([self.g1,self.g2],axis=1) # for permutation.
        t1=Groupinfo(self.g1)
        t2=Groupinfo(self.g2)    
        t1.metadistribution(x_grid,epsilon,bw,sigma,feat_info,log)
        t2.metadistribution(x_grid,epsilon,bw,sigma,feat_info,log)
        t1pos,t1eva=t1.summation(x_grid)
        t2pos,t2eva=t2.summation(x_grid)
        k=0
        diff=self.ShannonJanson(t1pos,t2pos,t1eva,t2eva,x_grid) ## original data's distance. 
        for i in tqdm(range(self.args.p_num)):
            z=np.take(z,np.random.permutation(z.shape[1]),axis=1,out=z)
            g1=z[:,d:]
            g2=z[:,:d]
            #print(g1.shape)
            t1=Groupinfo(g1)
            t2=Groupinfo(g2)
            t1.metadistribution(x_grid,epsilon,bw,sigma,feat_info,log)
            t2.metadistribution(x_grid,epsilon,bw,sigma,feat_info,log)
            t1pos,t1eva=t1.summation(x_grid)
            t2pos,t2eva=t2.summation(x_grid)
            dis=self.ShannonJanson(t1pos,t2pos,t1eva,t2eva,x_grid)
            #print("Distance for iteration{} {}".format(i+1,dis))
            k+=diff<dis
        return (k/self.args.p_num)
    
    def ShannonJanson(self,pos1,pos2,dis1,dis2,x_grid):
    ##returns sum of two KL divergence
        #cum1=self.simpson(dis1,x_grid)
        #cum2=simpson(dis2,x_grid)
        x=self.KLdivergence(pos1,pos2,dis1,dis2,x_grid)
        y=self.KLdivergence(pos2,pos1, dis2,dis1,x_grid)
        return x+y
    
    def KLdivergence(self,pos1,pos2,dis1,dis2,x_grid):
        pos_dis=0
        if pos1==0 or pos2==0:
            pos_dis=0
        else:
            pos_dis=pos1*np.log(pos1/pos2)
        
        con_sum=0
        stop_pos=len(x_grid)-1
        for i in range(len(x_grid)):
            if dis1[i]==0 or dis2[i]==0:
                stop_pos=i-1
                break
        g=dis1[:stop_pos]*(np.log(dis1[:stop_pos]/dis2[:stop_pos]))
        
        con_sum=self.simpson(g,x_grid[:stop_pos])
        return pos_dis+con_sum
    
    def simpson(self,dis,x_grid):
        con_sum=0
        con_sum+=dis[0]
        con_sum+=dis[len(dis)-1]
        for i in range(1,len(dis)-2):
            if i%2==0:
                con_sum+=2*dis[i]
            else:
                con_sum+=4*dis[i]
        h=(x_grid[-1]-x_grid[0])/len(x_grid)

        return con_sum*h/3
## if you do not hkavwe log-transformed data then transform it into log   
def datatransformation(w1):
    w2=copy.deepcopy(w1)
    n,d=np.shape(w2)
    if n==1:
         for j in range(d):
            if w2[0][j]==0:
                w2[0][j]=0
            else:
                w2[0][j]=math.log(w2[0][j])
    else:
        for i in range(n):
            for j in range(d):
                if w2[i][j]==0:
                    w2[i][j]=0
                else:
                    w2[i][j]=math.log(w2[i][j])
    w2=np.negative(w2)
    return w2


