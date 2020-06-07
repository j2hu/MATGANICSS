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
person_path='*******/GAN'

train_path=person_path+'/train/'

test_path=person_path+'/test/'

########
# this script provides per step of training as a default
#######

##############################################################

########################################################
# energy of per atom used to calcualte the binding enegy
E_Sn=-3.980911
E_S=-2.696603
E_Ca=-1.250692
E_O=-0.867783
###########################################################

############################################################
#parameter in the MatGAN

#about parameter
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)

#about pymatgen
patt_xrd = xrd.XRDCalculator('CuKa')

#about inputs of G and D:  crystal plane graph network, properties queue
global sample_num, rmat_num, series_num
sample_num=1 #output of G
rmat_num=28  #row nums of the matrix for the input of CNN 
series_num=3 #the number of the element in the queue (D)
#input of D


#about surface energy
global A
A =12.8282906600000004**2
################################################


######################################

#discriminator
class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.Dlstm=nn.LSTM(
            input_size=series_num,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out=nn.Sequential(
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.Sigmoid(),
        )
        #nn.Linear(32,1)
        #nn.Relu
        #nn.Linear
        #nn.Sigmoid
        
    def forward(self,x):
        D_out,(h_n,h_c)=self.Dlstm(x,None)
        out = self.out(D_out[:,-1,:]) #(batch,time step,input)   
        return out
####################################################


########################################################
#get energy from the OUTCAR
def get_total_energy(folder):
    energy_string=os.popen('grep TOTEN '+folder+'/OUTCAR | tail -1').read().split(' ')[-2]
    energy_slab=round(np.float64(float(energy_string)),5)
    return energy_slab

def get_binding_4O(E_t):
    E_binding= (E_t-6*E_Ca-4*E_Sn-10*E_S-4*E_O)/24
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


global extend_num, move_num
extend_num=1000
move_num=get_total_energy(train_path+'1000')
#print(move_num)

################################################



################################################
#1.extract the PXRD_data from a random poscar.(pymatgen)
#2.1 deal the PXRD peaks: the preprocessing of PXRD
#2.2 calculate out the L_matrix of C_P_G_N

# the baseline of pxrd
t1path=train_path+'1000/CONTCAR'
t1=mg.Structure.from_file(t1path)
t1pxrd=patt_xrd.get_pattern(t1)
global base_x,base_y
base_x=[]
base_y=[]
for i in range(len(t1pxrd)):
    if t1pxrd.y[i]>0.25 and t1pxrd.x[i]>20:
        base_x.append(t1pxrd.x[i])
        base_y.append(t1pxrd.y[i])

base_x=base_x[:28]
base_y=base_x[:28]

# randomly select the POSCAR/CONTCAR
def random_folder(file_path):
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
        if xrd_data4.y[i] >0.25  and xrd_data4.x[i]>20:
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
        
    xrd_x=abs(xrd_x-base_x)
    xrd_y=10*abs(xrd_y-base_y)
    
    xrd_x=np.sin(np.dot(1/180*np.pi,xrd_x))
    xrd_y=(np.arctan(xrd_y))/180*np.pi
    xrd_mat4.append(xrd_x)
    xrd_mat4.append(xrd_y)
    xrd_mat4=np.array(xrd_mat4)
    return xrd_mat4

##
################################
#def get_atoms_num(folder2):   #
#    xxx=tomgStructure(folder2)#
#    anum=len(xxx.sites)       # 
#    return anum               #
################################
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





###
##input_data_for_G
###
def GANs_Gmat(Random_Structure):
    global rmat_num
    RS_xrdmat = get_xrdmat(Random_Structure)
    multimat3_RS =  np.zeros((rmat_num,rmat_num),dtype='float32')
    multimat3_RS = np.asarray((np.dot(RS_xrdmat.T, RS_xrdmat)))
    return multimat3_RS

#this part is uesd to produce the input matrix for G<-Mat-GAN
###################################################################


###################################################################
#define the G<-Mat-GAN

class GNet(nn.Module):
    
    def __init__(self, input_size=(sample_num,28,28)):
        super(GNet, self).__init__()
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
#the initialzaiont of G and D
G1=GNet()
D1=DNet()

opt_D1=torch.optim.Adam(D1.parameters(),lr=0.01)
opt_G1=torch.optim.Adam(G1.parameters(),lr=0.01)

######
#users could load the the previous .pkl for training the G/D in the next step 
#G1.load_stat_dict(torch.load(person_path+'/GAN_G_step.pkl'))
#D1.load_stat_dict(torch.load(person_path+'/GAN_D_step.pkl'))
######

#intialization of input queue
train_series=[]
for i in range(series_num):
    path_s=random_xxpsk(train_path)
    ee1=get_total_energy(path_s)
    ee1=linear_transform(ee1)
    train_series.append(ee1)


##########################################################
#test                                                    #
#mg.Structure.from_file(train_path+'1000/CONTCAR')       #
##########################################################



mat_Gl=[]    #loss G
mat_Dl=[]    #loss D
pre_dd=[]    #out D
pre_gg=[]    #out G
error_test=[]
error_train=[]

###################################################################
###################################################################
# users could train Mat-GAN or load the .pkl
##################################################################
##################################################################
#per step in the training of Mat-GAN                     trianing#
##################################################################
##################################################################


#In the each folder of the dir_ of training or testing, there are a structrure file(POSCAR/CONTCAR) and a OUTCAR file(that is used to extract the properties data)

file_path=train_path
#tfset=[]  
for step in range(1,3):       


    sample_path=[]
    for i in range(1,sample_num+1):
        path_ = random_folder(file_path)
        sample_path.append(path_)

    X_DFT=0

    for subpath_ in sample_path:
    
        try:
            total_energy=get_total_energy(path_)
            X_DFT=linear_transform(total_energy)
        except:
            print(path1_)
         
        train_series.pop(-1)
        train_series.append(X_DFT)
    #update queue with X from D  
    input_series_D=np.asarray(train_series,dtype=np.float64)       
    input_series_D=Variable(torch.from_numpy(input_series_D[np.newaxis,np.newaxis,:]),requires_grad=True)
    
    Dout_real=D1(input_series_D)
    pre_dd.append(Dout_real.data.numpy().mean())
    
    G_input=[]
    for path2_ in sample_path:
        path2_=str(path2_)                
        
        try:
            file2pmg=tomgStructure(path2_)
            L_matrix=GANs_Gmat(file2pmg)
            
        except:
            pass
        G_input.append(L_matrix)
       
    G_input=np.asarray(G_input)
    G_input=G_input[np.newaxis,:,:,:] 
    G_input=np.asarray(G_input,dtype=np.float64) 
    G_input=Variable(torch.from_numpy(G_input),requires_grad=True)
    
    Gout=G1(G_input)
    Gout=round(Gout.data.numpy().mean(),6)   # properties by G

    #update queue with X from G
    train_series.append(Gout)
    train_series.pop(0)
        
    input_series_D=np.asarray(train_series,dtype=np.float64)       
    input_series_D=Variable(torch.from_numpy(input_series_D[np.newaxis,np.newaxis,:]),requires_grad=True)
    
    
    D_out_fake=D1(input_series_D)
    pre_gg.append(D_out_fake.data.numpy().mean())
    
    #loss
    D1_loss=-torch.mean(torch.log(D_out_real)+torch.log(1.-D_out_fake))
    dd=D1_loss.data.numpy().mean()
    mat_Dl.append(dd)
    
    G1_loss=torch.mean(torch.log(1.-D_out_fake))
    gg=G1_loss.data.numpy().mean()
    mat_Gl.append(gg)
    
    #------update Mat-GAN with loss 
    if step%2==0:
        opt_D1.zero_grad()
        D1_loss.backward(retain_graph=True)
        opt_D1.step()
        
        opt_G1.zero_grad()
        G1_loss.backward()
        opt_G1.step()
    else:
        opt_D1.zero_grad()
        D1_loss.backward()
        opt_D1.step()
    

    if step%2==0:
        print(step)
        print('error: ',abs(inverse_transform(Gout)-inverse_transform(X_DFT)))
        
        print(dd)
        print(gg)
        print(prob_Tfactor_mat0.data.numpy().mean())
        print(prob_G1_mat1.data.numpy().mean())
        
        
##############################################################################
##############################################################################
#statistics the performance of Mat-GAN in the training set and test set.   
    


def get_binding_4O(E_t):
    E_binding= (E_t-6*E_Ca-4*E_Sn-10*E_S-4*E_O)/24
    return E_binding


# In[30]:



E_Gibbs_test=[]
E_Gmodel_test=[]
abserrset=[]
MSEset=[]
err0set=[]
testfile2=[]
for m1,n1,fname in os.walk(test_path):
    for ieach in n1:
        ieach=test_path+ieach
        testfile2.append(ieach)
start=time.time()        
for path_ in testfile2:
    try:
        GGG=get_total_energy(path_)
        GGG=get_binding_4O(GGG)
        E_Gibbs_test.append(GGG)
        
        g_in=[]
        tomgS=tomgStructure(path_)
        gin=GANs_Gmat(tomgS)
        g_in.append(gin)
        g_in=np.asarray(g_in)
        g_in=g_in[np.newaxis,:,:,:]
        g_in=np.asarray(g_in,dtype=np.float64)
        g_in=Variable(torch.from_numpy(g_in),requires_grad=True)
        Gout=G1(g_in)
        G_data=Gout.data.numpy().mean()
        G_data=inverse_transform(G_data)
        G_data=get_binding_4O(G_data)
        E_Gmodel_test.append(G_data)
        #print(G_data)
        #print(GGG)
        abserr=abs(G_data-GGG)
        mse=(G_data-GGG)**2
        abserrset.append(abserr)
        MSEset.append(mse)
        err0=abs(abserr/GGG)
        err0set.append(err0)
    except:
        print(path_)
end=time.time()
print(end-start)


# In[31]:


print(np.asarray(abserrset).mean())

print(np.asarray(MSEset).mean())

#print(np.sqrt(np.asarray(MSEset).mean()))


# In[ ]:


print(abserrset)


# In[26]:


E_Gibbs_t=[]
E_Gmodel_t=[]
abs_t_errset=[]
err_t_0set=[]
tMSEset=[]
testfile=[]
for m1,n1,fname in os.walk(train_path):
    for ieach in n1:
        ieach=train_path+ieach
        testfile.append(ieach)


# In[25]:





# In[35]:


start=time.time()
#        
for path_ in testfile:
    try:
        GGG=get_total_energy(path_)
        GGG=get_binding_4O(GGG)

        E_Gibbs_t.append(GGG)
        g_in=[]
        tomgS=tomgStructure(path_)
        gin=GANs_Gmat(tomgS)
        g_in.append(gin)
        g_in=np.asarray(g_in)
        g_in=g_in[np.newaxis,:,:,:]
        g_in=np.asarray(g_in,dtype=np.float64)
        g_in=Variable(torch.from_numpy(g_in),requires_grad=True)
        Gout=G1(g_in)
        G_data=Gout.data.numpy().mean()
        G_data=inverse_transform(G_data)
        G_data=get_binding_4O(G_data)
        E_Gmodel_t.append(G_data)
        #print(G_data)
        #print(GGG)
        abserr=abs(G_data-GGG)
        tmse=(G_data-GGG)**2
        tMSEset.append(tmse)
        abs_t_errset.append(abserr)
        err0=abs(abserr/GGG)
        err_t_0set.append(err0)
    except:
        print(path_)
end=time.time()
print(end-start)



print(np.asarray(abs_t_errset).mean())

print(np.asarray(tMSEset).mean())




print(np.sqrt(np.asarray(tMSEset).mean()))




torch.save(G1.state_dict(),person_path+"/GAN_G_step.pkl") 
torch.save(D1.state_dict(),person_path+"/GAN_D_step.pkl")



