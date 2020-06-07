
# coding: utf-8

# In[1]:


import os
import re
import pymatgen as mg
import pymatgen.analysis.diffraction as anadi
import pymatgen.analysis.diffraction.xrd as xrd
import numpy as np
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import time


##############################################################
# define the dir to save the training smaples and test sample.i
# this .py script is for training the CSS_4O with Mat-GAN 
# the graph network<-structural descriptor--parttern A
# this script is used to reduplicate the results about Mat-GAN
# author complete the initial works in the jupyter notebook, we aslo advice users to perform the training of Mat-GAN in this envrionment.
# By our experiences, the interactive training process helps the users of the GAN.


#replace '****' with your path

person_path='*****/GCN'

train_path=person_path+'/train/'

test_path=person_path+'/test/'
########
# this script provides the training of GCN with L1-loss
#######


########################################################

########################################################
# energy of per atom used to calcualte the binding enegy
E_Sn=-3.980911
E_S=-2.696603
E_Ca=-1.250692
E_O=-0.867783
###########################################################

############################################################
#parameter in the GCN

#about torch
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)

#about pymatgen
patt_xrd = xrd.XRDCalculator('CuKa')


#about inputs of NN layers in GCN
global sample_num, rmat_num, series_num
sample_num=1 #output of G
rmat_num=28  #row nums of the matrix for the input of CNN 


########################################################
#get energy from the OUTCAR
def get_total_energy(folder):
    energy_string=os.popen('grep TOTEN '+folder+'/OUTCAR | tail -1').read().split(' ')[-2]
    energy_slab=round(np.float64(float(energy_string)),5)
    return energy_slab

def get_binding_4O(E_t):
    E_binding= (E_t-6*E_Ca-4*E_Sn-9*E_S-5*E_O)/24
    return E_binding

######################################################

#####################################################
#preprocessing
def linear_transform(energy):
    global extend_num, move_num
    energy_transform=(energy-move_num)*extend_num
    return energy_transform
def inverse_transform(energy_transform):
    global extend_num, move_num
    energy=energy_transform/extend_num+move_num
    return energy
'''
def get_energy_per_atom(energy):
    energy_per_atom=energy/atoms_num
    return energy_per_atom
'''
global extend_num, move_num
extend_num=10
move_num=round(get_total_energy(train_path+'2001'),2)
print(move_num)

################################################



################################################
#1.extract the PXRD_data from a random poscar.(pymatgen)
#2.1 deal the PXRD peaks: the preprocessing of PXRD
#2.2 calculate out the L_matrix of C_P_G_N

# the baseline of pxrd


#def get_basevalue():
#    t1path=train_path+'1000/CONTCAR'
#    t1=mg.Structure.from_file(t1path)
#    t1pxrd=patt_xrd.get_pattern(t1)
#    xxx=[]
#    yyy=[]
#    for i in range(len(t1pxrd)):
#        if t1pxrd.y[i]>0.1 and t1pxrd.x[i]>18:
#            xxx.append(t1pxrd.x[i])
#            yyy.append(t1pxrd.y[i])
#    xxx=xxx[:rmat_num]
#    yyy=yyy[:rmat_num]
#    return xxx,yyy



t1path=train_path+'2001/CONTCAR'
t1=mg.Structure.from_file(t1path)
t1pxrd=patt_xrd.get_pattern(t1)
global base_x,base_y
base_x=[]
base_y=[]
for i in range(len(t1pxrd)):
    if t1pxrd.y[i]>2 and t1pxrd.y[i]< 20:
        base_x.append(t1pxrd.x[i])
        base_y.append(t1pxrd.y[i])

base_x=base_x[:28]
base_y=base_x[:28]

# randomly select the POSCAR/CONTCAR
def random_xxpsk(file_path):
    folder=np.random.choice(glob.glob(file_path +"*"))
    #pos_name=folder+'/POSCAR'
    #out_name=folder+'/OUTCAR'
    return folder

# the step about Pymatgen POSCAR-> the format of pymatgen
def tomgStructure(folder):
    POSfile=folder+'/CONTCAR'      
    R_mgS=mg.Structure.from_file(POSfile)
    return R_mgS

###
##input_data_to_model
###

def get_xrdmat(mgStructure):
    global rmat_num
    xrd_data4 =patt_xrd.get_pattern(mgStructure)

    i_column = 28
    xxx=[]
    yyy=[]
    mat4=[]
    xrd_i=len(xrd_data4)
    for i in range(xrd_i):
        if xrd_data4.y[i] >2 and xrd_data4.y[i] <20:
            xxx.append(xrd_data4.x[i])
            yyy.append(xrd_data4.y[i])
    mat4.append(np.asarray(xxx))
    mat4.append(np.asarray(yyy))
    mat4=np.asarray(mat4)
    
    xrd_x=[]
    xrd_y=[]
    xrd_mat4=[]
    xrow=len(mat4[0])
    
    if xrow < i_column:
        for i in mat4[0]:
            xrd_x.append(i)
        for j in mat4[1]:
            xrd_y.append(j)
        for i in range(0,i_column-xrow):
            xrd_x.append(0)
            xrd_y.append(0)
        xrd_x=np.asarray(xrd_x)
        xrd_y=np.asarray(xrd_y)
    if xrow > i_column:
        xrd_x=mat4[0][:i_column]
        xrd_y=mat4[1][:i_column]
    if xrow == i_column:
        xrd_x= mat4[0]
        xrd_y= mat4[1]
        
    #xrd_x=abs(xrd_x-base_x)
    xrd_y=abs(xrd_y-base_y)/100
    
    xrd_x=10*np.sin(np.dot(1/180*np.pi,xrd_x))
    xrd_y=np.exp(np.sqrt(xrd_y))
    xrd_mat4.append(xrd_x)
    xrd_mat4.append(xrd_y)
    xrd_mat4=np.array(xrd_mat4)
    return xrd_mat4
###
##input_data_for_G
###
def GANs_Gmat(Random_Structure):
    global rmat_num
    RS_xrdmat = get_xrdmat(Random_Structure)
    multimat3_RS =  np.zeros((rmat_num,rmat_num),dtype='float32')
    multimat3_RS = np.asarray((np.dot(RS_xrdmat.T, RS_xrdmat)))
    return multimat3_RS
'''
#pattern ---F
#select one patter(AorF) for per training
t1path=train_path+'2001/CONTCAR'
t1=mg.Structure.from_file(t1path)
t1pxrd=patt_xrd.get_pattern(t1)
global base_x,base_y
base_x=[]
base_y=[]
for i in range(len(t1pxrd)):
    if t1pxrd.y[i]>2 and t1pxrd.y[i]< 20:
        base_x.append(t1pxrd.x[i])
        base_y.append(t1pxrd.y[i])

base_x=base_x[:28]
base_y=base_x[:28]




def get_xrdmat(mgStructure):
    global rmat_num
    xrd_data4 =patt_xrd.get_pattern(mgStructure)

    i_column = 28
    xxx=[]
    yyy=[]
    mat4=[]
    xrd_i=len(xrd_data4)
    for i in range(xrd_i):
        if xrd_data4.y[i] >2 and xrd_data4.y[i] <20:
            xxx.append(xrd_data4.x[i])
            yyy.append(xrd_data4.y[i])
    mat4.append(np.asarray(xxx))
    mat4.append(np.asarray(yyy))
    mat4=np.asarray(mat4)
    
    xrd_x=[]
    xrd_y=[]
    xrd_mat4=[]
    xrow=len(mat4[0])
    
    if xrow < i_column:
        for i in mat4[0]:
            xrd_x.append(i)
        for j in mat4[1]:
            xrd_y.append(j)
        for i in range(0,i_column-xrow):
            xrd_x.append(0)
            xrd_y.append(0)
        xrd_x=np.asarray(xrd_x)
        xrd_y=np.asarray(xrd_y)
    if xrow > i_column:
        xrd_x=mat4[0][:i_column]
        xrd_y=mat4[1][:i_column]
    if xrow == i_column:
        xrd_x= mat4[0]
        xrd_y= mat4[1]
        
    #xrd_x=abs(xrd_x-base_x)
    xrd_y=abs(xrd_y-base_y)/100
    
    xrd_x=10*np.sin(np.dot(1/180*np.pi,xrd_x))
    xrd_y=np.exp(np.sqrt(xrd_y))
    xrd_mat4.append(xrd_x)
    xrd_mat4.append(xrd_y)
    xrd_mat4=np.array(xrd_mat4)
    return xrd_mat4
'''



###################################################################
#define the GCN 
class GCNNet(nn.Module):
    def __init__(self, input_size=(sample_num,28,28)):
        super(GCNNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(#(3,28,28)
                in_channels=sample_num,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),#->(32,28,28)
            nn.ReLU(),#->(32,28,28)
            nn.MaxPool2d(kernel_size=2),
        )#->(#->(32,14,14))
        self.conv2=nn.Sequential(#->(32,14,14))
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),#->(64,14,14)
            nn.ReLU(),#->(64,14,14)
            nn.MaxPool2d(kernel_size=2),#->(64,7,7)
        )
        self.out=nn.Sequential(
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,sample_num),            
        )
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x) #batch(64,7,7)
        x=x.view(x.size(0),-1) #(batch, 64*7*7)
        output=torch.unsqueeze(self.out(x),dim=0)
        return output


#####################################################################
#the above parts are the module uesd to the process of initializaiton
#
#####################################################################

###############################################################
#the initialzaiont of G


G1= GCNNet()
opt_G1=torch.optim.Adam(G1.parameters(),lr=0.01)


######
#users could load the the previous .pkl for checking the MAE and MSE
#G1.load_stat_dict(torch.load(person_path+'/GCN_***.pkl'))
######

#In the each folder of the dir_ of training or testing, there are a structrure file(POSCAR/CONTCAR) and a OUTCAR file(that is used to extract the properties data)



trainfile=[]
for m1,n1,fname in os.walk(train_path):
    for ieach in n1:
        ieach=train_path+ieach
        trainfile.append(ieach)
i=0

loss_set=[]
for path_ in trainfile:
    X_DFT=[]
    try:
        total_energy=get_total_energy(path_)
        E_DFT=linear_transform(total_energy)
            #print(samp_Gibbs)
    except:
        print(path_)
            
    X_DFT.append(E_DFT)
        
    X_DFT=np.asarray(X_DFT,dtype=np.float64)       
    X_DFT=Variable(torch.from_numpy(X_DFT[np.newaxis,np.newaxis,:]),requires_grad=True)
    
    G_input=[]
           
     
    try:
        tomgS=tomgStructure(path_)
            #print(tomgS)
        L_matrix=GANs_Gmat(tomgS)
            #print(L_matrix)
    except:
        pass
    G_input.append(L_matrix)
       
    G_input=np.asarray(G_input)
    G_input=G_input[np.newaxis,:,:,:] 
    G_input=np.asarray(G_input,dtype=np.float64) 
    G_input=Variable(torch.from_numpy(G_input),requires_grad=True)
    
    Gout=G1(G_input)    
    
    G1_loss=torch.abs(torch.mean(Gout-float(E_DFT)))
    

    opt_G1.zero_grad()
    G1_loss.backward()
    opt_G1.step()
    
    i += 1
    loss_set.append(G1_loss)
    print(i,": ",G1_loss)


# In[ ]:


print(trainfile[6])


# In[15]:


torch.save(G1.state_dict(),"/home/hjj/Desktop/G1_GCN-CSS5O_10to80.pkl") 


# In[ ]:






# In[16]:


Eb_Gibbs_test=[]
Eb_Gmodel_test=[]
abserrsetb=[]
MSEsetb=[]
err0setb=[]
testfile=[]


# In[17]:


for m1,n1,fname in os.walk(test_path):
    for ieach in n1:
        ieach=test_path+ieach
        testfile.append(ieach)


# In[18]:



start=time.time()        
for path_ in testfile:
    try:
        GGG=get_total_energy(path_)
        GGG=get_binding_4O(GGG)
        

        Eb_Gibbs_test.append(GGG)
        
        G_input=[]
        tomgS=tomgStructure(path_)
        L_matrix=GANs_Gmat(tomgS)
        G_input.append(L_matrix)
        G_input=np.asarray(G_input)
        G_input=G_input[np.newaxis,:,:,:]
        G_input=np.asarray(G_input,dtype=np.float64)
        G_input=Variable(torch.from_numpy(G_input),requires_grad=True)
        Gout=G1(G_input)
        G_data=Gout.data.numpy().mean()
        G_data=inverse_transform(G_data)
        G_data=get_binding_4O(G_data)

        Eb_Gmodel_test.append(G_data)

        abserr=abs(G_data-GGG)
        mse=(G_data-GGG)**2
        abserrsetb.append(abserr)
        MSEsetb.append(mse)
        err0=abs(abserr/GGG)
        err0setb.append(err0)
    except:
        print(path_)
end=time.time()
print(end-start)


# In[19]:


print(np.asarray(abserrsetb).mean())

print(np.asarray(MSEsetb).mean())


# In[20]:


X_DFT_testb=[]
E_Gmodel_testb=[]
abs_t_errsetb=[]
err_t_0setb=[]
tMSEsetb=[]
testfileb=[]
for m1,n1,fname in os.walk(train_path):
    for ieach in n1:
        ieach=train_path+ieach
        testfileb.append(ieach)

start=time.time()        
for path_ in testfileb:
    try:
        GGG=get_total_energy(path_)
        GGG=get_binding_4O(GGG)
        X_DFT_testb.append(GGG)
        
        G_input=[]
        tomgS=tomgStructure(path_)
        L_matrix=GANs_Gmat(tomgS)
        G_input.append(L_matrix)
        G_input=np.asarray(G_input)
        G_input=G_input[np.newaxis,:,:,:]
        G_input=np.asarray(G_input,dtype=np.float64)
        G_input=Variable(torch.from_numpy(G_input),requires_grad=True)
        Gout=G1(G_input)
        G_data=Gout.data.numpy().mean()
        G_data=inverse_transform(G_data)
        G_data=get_binding_4O(G_data)
        E_Gmodel_testb.append(G_data)
        #print(G_data)
        #print(GGG)
        abserr=abs(G_data-GGG)
        mse=(G_data-GGG)**2
        abs_t_errsetb.append(abserr)
        tMSEsetb.append(mse)
        err0=abs(abserr/GGG)
        err_t_0setb.append(err0)
    except:
        print(path_)
end=time.time()
print(end-start)



print(np.asarray(abs_t_errsetb).mean())

print(np.asarray(tMSEsetb).mean())


torch.save(G1.state_dict(),person_path+"/GCN.pkl") 

