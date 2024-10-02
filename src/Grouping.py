import numpy as np
from scipy.stats import gaussian_kde
import copy
from src.Featuring import Feature
import math
class Groupinfo():
    ## X will be d*n matrix where d is number of feature and n is number of sample
    def __init__(self,X):
        self.X=X
        self.feature_distribution=[]
        self.feature_posterior=[]

    def getsamplenum(self):
        n,d=np.shape(self.X)
        num_lack=[]
        for i in range(n):
            n_zeros = np.count_nonzero( self.X[i,:])
            num_lack.append(n_zeros)
        return np.array(num_lack)
    
    def getsigma(self):
        n,d=np.shape(self.X)
        sigma=[]
        for i in range(n):
            sigma.append(np.var(self.X[i,:]))
        return sigma
    
    def metabandwidth(self,sigma,sigmas,bw,log=False):
        if log == False:
            return math.exp(0.1278*(sigma**2)-0.587*sigma+0.9244) ## our predefined values. 
        else:
            model = np.poly1d(np.polyfit(sigmas, bw, 2))
            coef=model.coefficients
            return ((sigma**2)*coef[0]+(sigma)*coef[1]+coef[2])
        
    def finddistribution(self,x_grid,epsilon,feat_info):
        ## each feature for each group will have a information for bandwidth and posterior probability for 0 occurences..
        n,d=np.shape(self.X)
        group_bandwidth=[]
        group_posterior=[]
        for i in range(n):
            bootstrapnum=100 ##will be a hyperparameter..    
            temp=Feature(i,self.X[i,:],bootstrapnum,epsilon,feat_info )
            group_bandwidth.append(temp.findBandwidth(x_grid))
            group_posterior.append(temp.findPosterior())
        self.group_bandwidth=group_bandwidth
        self.group_posterior=group_posterior

    def metadistribution(self, x_grid, epsilon,bw,sigma,feat_info, log=False):
        n,d=np.shape(self.X)
        group_bandwidth=[]
        group_posterior=[]
        for i in range(n):
            bootstrapnum=100
            temp=Feature(i,self.X[i,:],bootstrapnum,epsilon,feat_info)
            group_posterior.append(temp.findPosterior())
        sigmas=self.getsigma()  
        
        self.group_posterior=group_posterior
        for s in sigmas:
            if s !=0:
                group_bandwidth.append(self.metabandwidth(math.sqrt(s),sigma,bw,log))
            else:
                group_bandwidth.append(0)
        self.group_bandwidth=group_bandwidth

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
    
    def get_feature_distribution(self):
        try:
            getattr(self, "group_bandwidth")
        except AttributeError:
            raise RuntimeError("You must get the information of distribution for each feature before getting the distribution")
        n,d=np.shape(self.X)
        feature_distribution=[]
        for i in range(n):
            cur_data=self.X[i,:]
            if self.group_bandwidth[i]==0: #which means that the sigma is zero. 
                feature_distribution.append(0)

                continue
            else:
                cur_data=cur_data.tolist()
                kde = gaussian_kde(cur_data, bw_method=self.group_bandwidth[i] )
                feature_distribution.append(kde)
        return self.group_posterior, feature_distribution, self.group_bandwidth
    
    def summation(self,x_grid):
        ##must be done after finddistribution
        try:
            getattr(self, "group_bandwidth")
        except AttributeError:
            raise RuntimeError("You must get the information of distribution for each feature before summation")
        ## for each obtained distribution we need to get the posterior assuming uniform for each feature 
        ## posterior ~   p(x|i)*pi(i) p(x) is a distribution at i'th feature, and pi(i) we assume usniform
        n,d=np.shape(self.X)
        # print(self.group_posterior)
        summedposterior=np.mean(self.group_posterior)
        ##evaluation means averaged(in this case) likelihood for the whole distribution.
        evaluation=0
        for i in range(n):
            cur_data=self.X[i,:]
            if self.group_bandwidth[i]==0:
                continue
            else:
                cur_data=cur_data.tolist()
                kde = gaussian_kde(cur_data, bw_method=self.group_bandwidth[i] )
                ## note that each feature will have corresponding bandwidth and KDE estimation for the whole group..
                p=kde.evaluate(x_grid)

                cum=kde.integrate_box_1d(0,20)
                #print("cum:{}".format(cum))
                evaluation=evaluation+(1-self.group_posterior[i])*p/(cum*n)
        return summedposterior,evaluation  