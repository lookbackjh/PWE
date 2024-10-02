
import numpy as np
from scipy.stats import gaussian_kde
import math
from sklearn.utils import resample
class Feature():

    def __init__(self,feat_num,info,bootstrapnum,epsilon,feat_info) -> None:
        self.feat_num=feat_num
        self.info=info ## an 1*n matrix for each group.
        self.bootstrapnum=bootstrapnum ##hyperparameter
        self.epsilon=epsilon
        self.feat_info=feat_info

    def findBandwidth(self,x_grid):
        k=0
        #print(len(self.info))
 
        if len(self.info[self.info!=0])!=0:
            OOB,bootstrap=self.doBootstrap(self.info[self.info!=0],self.bootstrapnum) ## doing bootstrap, make both OOB and Bootstrap samples (excluding zeros)  
            self.bandwidth=self.AndersonDarling(bootstrap,OOB,x_grid,self.epsilon )## via Anderson-Darling, Obtain the bandwidth
            #print("AD time:{}".format(time))         
            k=self.bandwidth
        return k
        
    def findPosterior(self):
        ## With Bayesian processm obtain posterior mean for probability of zero occurence 'theta'
        #print(len(self.info))
        p_zeros = np.count_nonzero( self.info==0)/len(self.info) ## a probability of zero occurence in group(used as a mean for likelihood)

        prior=self.feat_info[self.feat_num]
        if prior>1: ## if there is no information for the prior
            return p_zeros ##just return a probability of zero occurence in group. 
        estimated_sigma=self.getBootstrapsigma(self.info,self.bootstrapnum) ## sigma for likelihood obtained via bootstrapping
        self.post_mean=self.getposterior(p_zeros,self.feat_num,estimated_sigma,self.feat_info) ## using likelihood above and known prior( with estimated interval), obtain posterior mean for zero occurence.
        return self.post_mean

    def AndersonDarling(self,bootstrapsample,OOB,x_grid,epsilon):
        ## get KDE estimation for each bootstrap...
        #start=time.time()
        cur_data=bootstrapsample[0]
        S=0
        bandwidth_loss=[]
        MIN_INT=-999999999
        cur_min=MIN_INT
        bandwidths=np.arange(0.1,9,0.1)
        opt=0
        ## interation : for every bandwith, for length of Bootstrap number , for length of OOB number for each bootstrap, integration( another interation needed) 
        for j in bandwidths:
            S=0
            for k in range(len(OOB)):
                cur_data=bootstrapsample[k]
                cur_data=cur_data+0.00001*np.random.rand(len(cur_data))## little noise to avoid error
                cur_OOB=OOB[k]
                cur_data=cur_data.tolist()
                if len(cur_data)==1 :
                    continue
                if len(cur_OOB)==0:
                    continue
                else:
                    kde =gaussian_kde(cur_data,bw_method=j)
                np.sort(cur_OOB)
                cumu_info=[]
                for i in range(len(cur_OOB)):
                    cumu=kde.integrate_box_1d(x_grid[0], cur_OOB[i]) ##  중복을 방지하기 위해 미리 계산 
                    cumu_info.append(cumu)
                for i in range(len(cur_OOB)): ##must be in increasing order.
                    ##the range could be hyperparameter..
                    cur_lnF=np.log(epsilon) ## when integration results in 0 
                    cumu=cumu_info[i]
                    if cumu>=epsilon:
                        cur_lnF=np.log(cumu)
                    inversecumu=1-cumu_info[len(cur_OOB)-1-i]
                    cur_lnTail_F=math.log(1-epsilon) ## when integration results in 1
                    if inversecumu>=epsilon:
                        cur_lnTail_F=math.log(inversecumu)
                    S+=(2*i+1)*(cur_lnF+cur_lnTail_F)/len(cur_OOB)     ##Anderson-Darling formation without zeros.
            if(S-len(cur_OOB))>cur_min:
                cur_min=S-len(cur_OOB)
                ## the smaller the anderson-darling, the better it its to kernel.
                opt=j
        #end=time.time()
        #print("AD Tim:{}".format(end-start))
        return opt ##returns bandwidth that gives smallest bandwidth

    def getposterior(self,p_zeros,feat_num,estimated_sigma,feat_info):
    ##returns estimated mean for the posterior
        hyper_sigma=0.3 ##hyperparameter.
        ##poesterior ~ prior * likelihood/marginal , prior as our known belief and setting interval  as a hyperparametr, and likelihood with sigma obtained by bootstrapping 
        prior=feat_info[feat_num]
        if prior>1: ## if there is no information for the prior
            prior=p_zeros
        posterior_mean= (prior*estimated_sigma**2+p_zeros*hyper_sigma**2)/(estimated_sigma**2+hyper_sigma**2) #mathematicall operation 
        #print("featnumber:{}' posterior :{}".format(feat_num,posterior_mean))
        return posterior_mean

    def doBootstrap(self,X,bootstrap_num):
    ## returns OOB and Bootstrapped data
        bootstrapsample=[]
        OOB=[]
        sample_num=len(X) ##hyperparam.
        for i in range(bootstrap_num):
            bootstrapsample.append(resample(X, n_samples=sample_num, replace=True)) ##bootstrapping
            OOB.append(np.setdiff1d(X,bootstrapsample[i]))
        return OOB,bootstrapsample
    
    def getBootstrapsigma(self,X,bootstrap_num):
    ##returns estimatedsigma for the likelihood of the distribuion of /theta   
        p_zeros=[] ##probability of certain feature having zero counts.
        bootstrapsample=[]
        sample_num=len(X)
        if np.count_nonzero( X==0)==0: ##if feature in group does not contain 0 
            return 0
        else:
            for i in range(bootstrap_num):
                bootstrapsample.append(resample(X, n_samples=sample_num, replace=True)) ##bootstrapping
                p_zeros.append(np.count_nonzero( bootstrapsample[i]==0)/sample_num)
        return np.var(p_zeros)