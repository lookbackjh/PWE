from src.Util import Permutator
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import time
from scipy.stats import truncnorm
class Sensitivity():

    def __init__(self,args) -> None:
        self.args=args
        pass

    def simulate(self,feat_info):
        if self.args.task==0:        ## simulation for sparse ratio
            k_candidate=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            d = defaultdict(list) ## a dictionary to get the results for the sd and mean for the p-value (too obtain the standard deviation and mean for the p-value repeating 50 times of 1000 permutation)
            for k in k_candidate:
                pvals=[]
                print(k)
                for i in tqdm(range(self.args.repeat_num)):
                    p1_100,p2_100=self.get_group_sparse(self.args.num_feature,k)
                    p=Permutator(p1_100,p2_100,feat_info,self.args)
                    pval=p.metapermutation()
                    pvals.append(pval)
                    d[k]=pvals
            for k in k_candidate:
                print("for differnce sparse ratio k: {}, mean of the p-value :{} sd of the p-value :{}".format(k,np.mean((np.array(d[k]))), np.sqrt(np.var(np.array(d[k])))))
        else:
            pvals=[]
            for i in tqdm(range(self.args.repeat_num)):
                p1_100,p2_100=self.get_group_nk(self.args.num_feature,self.args.same_ratio)
                p=Permutator(p1_100,p2_100,feat_info,self.args)
                pval=p.metapermutation()
                pvals.append(pval)
            print("For feature number {} and Differnt Ratio {}".format(self.args.num_feature,self.args.same_ratio))    
            print("Mean of the P-value:{} Standard Deviation:{}".format(np.mean(np.array(pvals)),np.std(np.array(pvals))))

    def get_group_nk(self,n,k):
        bounds=[0,20]
        pointwiseentropyg1=[]
        pointwiseentropyg2=[]
        ratio=k*0.01
        predecsion_mean=np.random.randint(1,13,int(ratio*n))
        predecision_sd=np.random.randint(1,6,int(ratio*n))
        diff_mean1=np.random.randint(1,13,int((1-ratio)*n))
        diff_mean2=np.random.randint(1,13,int((1-ratio)*n))
        diff_sig1=np.random.randint(1,6,int((1-ratio)*n))
        diff_sig2=np.random.randint(1,6,int((1-ratio)*n))
        for i in range(int(ratio*n)):
            loc=predecsion_mean[i]
            scale=predecision_sd[i]
            sample1=[]
            sample2=[]
            for sample in range(20):
                s1=truncnorm.rvs((bounds[0]-loc)/scale, (bounds[1]-loc)/scale, loc=loc, scale=scale)
                s2=truncnorm.rvs((bounds[0]-loc)/scale, (bounds[1]-loc)/scale, loc=loc, scale=scale)
                sample1.append(s1)
                sample2.append(s2)
            pointwiseentropyg1.append(sample1)
            pointwiseentropyg2.append(sample2)
        for i in range(int((1-ratio)*n)):
            loc1=diff_mean1[i]
            scale1=diff_sig1[i]
            loc2=diff_mean2[i]
            scale2=diff_sig2[i]
            sample1=[]
            sample2=[]
            for sample in range(20):
                s1=truncnorm.rvs((bounds[0]-loc1)/scale1, (bounds[1]-loc1)/scale1, loc=loc1, scale=scale1)
                s2=truncnorm.rvs((bounds[0]-loc2)/scale2, (bounds[1]-loc2)/scale2, loc=loc2, scale=scale2)
                sample1.append(s1)
                sample2.append(s2)
            pointwiseentropyg1.append(sample1)
            pointwiseentropyg2.append(sample2)

        p1_10=np.array(pointwiseentropyg1)
        p2_10=np.array(pointwiseentropyg2)
        return p1_10,p2_10
    
    def get_group_sparse(self,n,sparseratio): 
        bounds=[0,20]
        predecsion_mean=np.random.randint(1,13,n)
        predecision_sd=np.random.randint(1,6,n)
        pointwiseentropyg1=[]
        pointwiseentropyg2=[]
        np.random.normal(3,1,20)
        predecsion_mean=np.random.randint(1,13,int(n))
        predecision_sd=np.random.randint(1,6,int(n))
        ##  groups that will be generated with  with same sparse ratio.. 
        for i in range(int(sparseratio*n)):
            loc=predecsion_mean[i]
            scale=predecision_sd[i]
            sample1=[]
            sample2=[]
            sparsity=0.1*np.random.randint(0,10)
            for sample in range(int(20*(1-sparsity))):
                s1=truncnorm.rvs((bounds[0]-loc)/scale, (bounds[1]-loc)/scale, loc=loc, scale=scale)
                s2=truncnorm.rvs((bounds[0]-loc)/scale, (bounds[1]-loc)/scale, loc=loc, scale=scale)
                sample1.append(s1)
                sample2.append(s2)
            for sample in range(20-int(20*(1-sparsity))):
                sample1.append(0)
                sample2.append(0)
            pointwiseentropyg1.append(sample1)
            pointwiseentropyg2.append(sample2)

        ##  groups that will be generated with  with same sparse ratio.. 
        for i in range(int((1-sparseratio)*n)):
            loc=predecsion_mean[i+int(sparseratio*n)]
            scale=predecision_sd[i+int(sparseratio*n)]
            sample1=[]
            sample2=[]
            sparsity1=0.1*np.random.randint(1,5)
            sparsity2=0.1*np.random.randint(5,10)
            for sample in range(int(20*(1-sparsity1))):
                s1=truncnorm.rvs((bounds[0]-loc)/scale, (bounds[1]-loc)/scale, loc=loc, scale=scale)
                sample1.append(s1)
            for sample in range(20-int(20*(1-sparsity1))):
                s1=truncnorm.rvs((bounds[0]-loc)/scale, (bounds[1]-loc)/scale, loc=loc, scale=scale)
                sample1.append(0)
            for sample in range(int(20*(1-sparsity2))):
                s2=truncnorm.rvs((bounds[0]-loc)/scale, (bounds[1]-loc)/scale, loc=loc, scale=scale)
                sample2.append(s2)
            for sample in range(20-int(20*(1-sparsity2))):
                s2=truncnorm.rvs((bounds[0]-loc)/scale, (bounds[1]-loc)/scale, loc=loc, scale=scale)
                sample2.append(0)
            pointwiseentropyg1.append(sample1)
            pointwiseentropyg2.append(sample2)
        p1_10=np.array(pointwiseentropyg1)
        p2_10=np.array(pointwiseentropyg2)
        return p1_10,p2_10