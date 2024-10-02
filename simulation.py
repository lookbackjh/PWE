import argparse
import pandas as pd
import math
import numpy as np
import matplotlib.pylab as plt
import numpy as np
import argparse
from src.Senstivity import Sensitivity
import copy
import math
from tqdm import tqdm
from src.Util import Permutator
import time
from tqdm import tqdm
import os
#import dirichlet
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Args Sparse Simulation')
parser.add_argument('--p_num', type=int, default=1000, help='num ber of permutation')
parser.add_argument('--x_grid_start',type=int, default=1,help='start of the x-axis for integration')
parser.add_argument('--x_grid_end',type=int, default=20,help='end of the x-axis for integration')
parser.add_argument('--interval',type=float,default=0.1,help='interval for x-axis' )
parser.add_argument('--epsilon',type=float,default=0.0001,help='to_avoid zero division')
parser.add_argument('--repeat_num',type=int, default=1, help='to see the mean and sd for the p-value after repeated permutation')
parser.add_argument('--predefined',type=bool,default=False,help='False if there is predefined meta analysis')
parser.add_argument('--theta',type=float,default=0.0001,help='theta value for dirichlet')
parser.add_argument('--num_samples',type=int,default=50,help='number of samples')
parser.add_argument('--scenario',type=int,default=1,help='scenario number')
args, _ = parser.parse_known_args()



def seed_everything(seed):
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    # torch.use_deterministic_algorithms(True)



seed=1234
seeds=[10]
betas=[0.9]

seed_everything(seed)
if args.scenario==1: # scenario 1 for top 0-10 selection
    start_idx=0
    end_idx=10
    sort_option=True
elif args.scenario==2: ## scenario 2 for top 100-200 selection
    start_idx=100
    end_idx=200
    sort_option=True
else: ## scenario 3 for random 300 selection
    start_idx=0
    end_idx=300
    sort_option=False


otu=pd.read_csv("Example/Meta_Analysis_Example/throat_otu.csv") ## this is the throat microbiome data, refer to LDM github. https://github.com/yijuanhu/LDM

# drop the first column
otu=otu.drop(columns=['Unnamed: 0'])

#dive each row by sum of the row
otu=otu.div(otu.sum(axis=1), axis=0)


ratio=otu.mean()

# drop three most abundant species
ratio=ratio.sort_values(ascending=False)
ratio=ratio.iloc[3:]

# randomly select the microbiomes to change the ratio
ratio=ratio.to_numpy()
if sort_option ==False:
    # sort ratio
    np.random.shuffle(ratio)



import copy

ratio_fixed_1=copy.deepcopy(ratio[:start_idx])
ratio_to_change=copy.deepcopy(ratio[start_idx:end_idx])
#ratio_to_change
ratio_fixed_2=copy.deepcopy(ratio[end_idx:])

decreased=ratio_to_change[:int(len(ratio_to_change)/2)]-ratio_to_change[:int(len(ratio_to_change)/2)]*beta

ratio_change_first_half=ratio_to_change[:int(len(ratio_to_change)/2)]-decreased
ratio_second_half=copy.deepcopy(ratio_to_change[int(len(ratio_to_change)/2):])
increased=np.zeros(len(ratio_second_half))
sum_ratio_second_half=ratio_second_half.sum()   

for i in range(len(ratio_second_half)):

    additive=(decreased.sum()*0.5)/len(ratio_second_half)
    proportion=(decreased.sum()*0.5)*ratio_second_half[i]/sum_ratio_second_half
    increased[i]=additive+proportion

# distribute the increased ratio to the second half with equivalent to 

ratio_second_half=ratio_second_half+increased

g1_param=copy.deepcopy(ratio)
g_first_half=np.concatenate((ratio_fixed_1,ratio_change_first_half,ratio_second_half))
g2_param=np.concatenate((g_first_half,ratio_fixed_2))


theta_2=0.0001 # overdispersion parameter


overdispersion=(1-theta_2)/theta_2

g1=np.random.dirichlet(g1_param*(1/overdispersion),size=args.num_samples)
g2=np.random.dirichlet(g2_param*(1/overdispersion), size=args.num_samples)

#create count poisson
g1_count_total=np.zeros(args.num_samples)
g2_count_total=np.zeros(args.num_samples)


#what i watn to do is get dividing rowwise summation
# g1_normalized = g1_count / g1_count.sum(axis=1)[:, np.newaxis]
# g2_normalized = g2_count / g2_count.sum(axis=1)[:, np.newaxis]
g1_count=np.zeros((args.num_samples,len(g1_param)))
g2_count=np.zeros((args.num_samples,len(g1_param)))
g1_count_df=pd.DataFrame(g1_count)
g2_count_df=pd.DataFrame(g2_count)


#randomly create integer from 2000 to 10000 with the number of samples
sample_counts=np.random.randint(10000,15000,args.num_samples)
#sample_counts=np.random.poisson(10000,args.num_samples)

for i in range(args.num_samples):
    g1_count_total[i]=np.random.poisson(sample_counts[i])
    g2_count_total[i]=np.random.poisson(sample_counts[i])


for i in range(args.num_samples):
    g1_count[i]=np.random.multinomial(g1_count_total[i],g1[i])
    g2_count[i]=np.random.multinomial(g2_count_total[i],g2[i])


#g1_count_df['group']='g1'
#g2_count_df['group']='g2'
df=pd.concat([g1_count_df,g2_count_df])



df=df.reset_index(drop=True)
filename=f"s150/S1_start_idx_{start_idx}_end_idx_{end_idx}_num_samples_{args.num_samples}_theta_{theta_2}_beta_{beta}_seed{seed}.csv"
#filename=f"S1{end_idx-start_idx}_p_num_{args.p_num}_seed{seed}_beta{beta}.csv"
#df.to_csv(filename,index=False)


g1_normalized = g1_count / g1_count.sum(axis=1)[:, np.newaxis]
g2_normalized = g2_count / g2_count.sum(axis=1)[:, np.newaxis]


#create dataframe with g1_normalized and g2_normalized
# concatenate g1_normalized and g2_normalized
# please make into single dataframe with new column indexing group number

g1_normalized_df=pd.DataFrame(g1_normalized)
g2_normalized_df=pd.DataFrame(g2_normalized)

# g1_normalized_df['group']=1
# g2_normalized_df['group']=2

# df=pd.concat([g1_normalized_df,g2_normalized_df])
# df=df.reset_index(drop=True)
# df.to_csv("Example/Meta_Analysis_Example/simulated_data_09.csv",index=False)

epsilon = 1e-10  # Small constant to avoid log(0)
g1_entropy = np.where(g1_normalized > 0.0, -np.log(g1_normalized ), 0)
g2_entropy = np.where(g2_normalized > 0.0, -np.log(g2_normalized ), 0)

g1_entropy = g1_entropy.T
g2_entropy = g2_entropy.T
#drop columns where all values are zero
g1_zero_columns=np.where(~g1_entropy.any(axis=1))[0]
g2_zero_columns=np.where(~g2_entropy.any(axis=1))[0]
#combine two column index
zero_columns=np.concatenate((g1_zero_columns,g2_zero_columns))
g1_entropy=np.delete(g1_entropy,zero_columns,axis=0)
g2_entropy=np.delete(g2_entropy,zero_columns,axis=0)

pvals=[]

x_grid=np.arange(0,17,0.1)
feat_info=np.zeros(g1_entropy.shape[0])
feat_info+=1.1 ## as this is simulated  case, you cannot consider the prior-posterior case. 
args.num_feature=g1_entropy.shape[0]

simulator=Sensitivity(args)

    #p1_100,p2_100=simulator.get_group_sparse(args.num_feature,k)
till_100=g1_entropy[0:100]
from_100_to_200=g1_entropy[100:200]
from_200_to_full=g1_entropy[200:]
unselected_g1=np.concatenate((till_100,from_200_to_full))
till_100=g2_entropy[0:100]
from_100_to_200=g2_entropy[100:200]
from_200_to_full=g2_entropy[200:]
unselected_g2=np.concatenate((till_100,from_200_to_full))



p=Permutator(unselected_g1,unselected_g2,feat_info,args)
pval=p.metapermutation()
print(pval)
#save the p-valueS


