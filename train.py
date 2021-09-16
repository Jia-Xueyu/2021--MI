from classifier_resnet import *
from preprocess_data import MIData
import numpy as np
import os
from torch.utils.data import DataLoader,TensorDataset
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchsummary import summary
import argparse
from torch.autograd import Variable
from tqdm import tqdm
import random
import math
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt

gpus=[0]
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))

crop_time = 6
srate = 250


def train(opt):

    data=MIData(opt.path,opt.subjects,opt.block_num)
    data=data.crop_data()

    train_subject_data=[i for i in data if i['subject'] in [j+1 for j in range(4)]]
    test_subject_data=[i for i in data if i['subject']==5]
    train_concate_augment_data=data_augmentation_by_concat(train_subject_data)
    test_concate_augment_data=data_augmentation_by_concat(test_subject_data)

    train_overlap_data = [np.expand_dims(i['data'], 0) for i in train_subject_data]
    train_overlap_data=np.concatenate(train_overlap_data,0)
    # for i in range(data.shape[0]):
    #     data[i,0:64,:]=preprocessing.scale(data[i,0:64,:],axis=1)
    train_overlap_augment_data=[]
    for i in range(train_overlap_data.shape[0]):
        train_overlap_augment_data.append(data_augmentation_by_overlap(train_overlap_data[i]))

    train_overlap_augment_data=np.concatenate(train_overlap_augment_data,0)


    test_overlap_data = [np.expand_dims(i['data'], 0) for i in test_subject_data]
    test_overlap_data=np.concatenate(test_overlap_data,0)
    # for i in range(data.shape[0]):
    #     data[i,0:64,:]=preprocessing.scale(data[i,0:64,:],axis=1)
    test_overlap_augment_data=[]
    for i in range(test_overlap_data.shape[0]):
        test_overlap_augment_data.append(data_augmentation_by_overlap(test_overlap_data[i]))

    test_overlap_augment_data=np.concatenate(test_overlap_augment_data,0)

    train_augment_data=np.concatenate([train_concate_augment_data,train_overlap_augment_data],0)
    test_augment_data=np.concatenate([test_concate_augment_data,test_overlap_augment_data],0)
    # data=slice_data(data)
    print(train_augment_data.shape)
    print(test_augment_data.shape)
    train_data=train_augment_data[:,0:64,:]
    train_label=train_augment_data[:,64,0]
    train_data=np.expand_dims(train_data,1)
    test_data=test_augment_data[:,0:64,:]
    test_label=test_augment_data[:,64,0]
    test_data=np.expand_dims(test_data,1)
    # train_data,test_data,train_label,test_label=train_test_split(train_data,train_label,test_size=0.2,shuffle=True)
    train_data,train_label=torch.from_numpy(train_data),torch.from_numpy(train_label)
    test_data = torch.from_numpy(test_data).type(torch.cuda.FloatTensor)
    dataset=TensorDataset(train_data,train_label)
    dataloader=DataLoader(dataset,batch_size=opt.batch_size,shuffle=True)
    model=ResNet50()

    model=model.cuda()
    model=nn.DataParallel(model,device_ids=[i for i in range(len(gpus))]).cuda()
    summary(model, (1, 64, 1750))
    optimizer=torch.optim.Adam(model.parameters(),lr=opt.lr)
    criterion=nn.CrossEntropyLoss().cuda()


    for e in range(opt.epochs):
        train_loss=0
        model.train()
        pbar=tqdm(enumerate(dataloader),total=len(dataloader))
        for i, (t_data,t_label) in pbar:
            t_data=Variable(t_data.cuda().type(torch.cuda.FloatTensor))
            t_label=Variable(t_label.cuda().type(torch.cuda.LongTensor))
            pred=model(t_data)
            loss=criterion(pred,t_label)
            train_loss+=loss.detach().cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('epoch : {}, train loss: {:.3f}'.format(e,train_loss/len(dataset)))

        model.eval()
        temp=model(test_data)
        pred_test=torch.max(temp,1)[1]
        test_acc=(pred_test.cpu().numpy()==test_label).sum()/len(test_label)
        print('test acc: ',test_acc)
    torch.save(model.module,'resnet50.pth')


def slice_data(data):
    offset=random.random()
    slice_time=random.randint(0,7)
    offset_point=int(offset*srate)
    print('offset: ',offset,' slice_time: ',slice_time)
    return data[:,:,offset_point:offset_point+int(slice_time)*srate]

def data_augmentation_by_concat(data):#相同label不同位置随机切块拼接,data is a dict
    expansion_factor=50
    trial_num=3
    class_num=3
    concate_slice_num=4
    subject_num=np.unique([i['subject'] for i in data]).tolist()
    block_num=np.unique([i['block'] for i in data]).tolist()
    data_slice_all_length=math.floor(data[0]['data'].shape[1]/concate_slice_num)
    crop_slice_length=math.floor(crop_time*srate/concate_slice_num)
    augmentation_data=[]
    for i in range(len(subject_num)):
        for j in range(expansion_factor):
            temp_data=[{'label':t['data'][64,0],'data':t['data']} for t in data if t['subject']==subject_num[i]]
            temp_data=sorted(temp_data,key=lambda k:k['label'])
            label_start_index=[]
            for tt in range(3):
                label_start_index.append([t['label'] for t in temp_data].index(tt))

            temp_data=[t['data'] for t in temp_data]


            temp_label=[[],[],[]]
            for index_label in range(len(temp_label)):
                if index_label==2:
                    seq=[t for t in range(label_start_index[index_label],15*3)]
                else:
                    seq=[t for t in range(label_start_index[index_label],label_start_index[index_label+1])]
                choice_block = random.sample(seq, concate_slice_num)
                for index_block in range(len(choice_block)):
                    start = random.randint(0, data_slice_all_length - crop_slice_length - 1) + index_block * crop_slice_length
                    if index_block==len(choice_block)-1:
                        temp_length=0
                        for ii in range(len(temp_label[index_label])):
                            temp_length+=temp_label[index_label][ii].shape[1]
                        temp_length=crop_time*srate-temp_length
                        end=start+temp_length
                    else:
                        end = start + crop_slice_length
                    temp_label[index_label].append(temp_data[choice_block[index_block]][:,start:end])
                temp_label[index_label]=np.concatenate(temp_label[index_label],1)
            augmentation_data.extend(temp_label)

            # for index_block in range(len(choice_block)):
            #     for index_label in range(3):
            #         start=random.randint(0,data_slice_all_length-crop_slice_length-1)+index_block*crop_slice_length
            #         # if index_block==len(choice_block)-1:
            #         #     end=start+crop_time*srate-(len(choice_block)-1)*crop_slice_length
            #         # else:
            #         end=start+crop_slice_length
            #
            #         for ii in range(choice_block[index_block]*3,(choice_block[index_block]+1)*3):
            #             if temp_data[ii][64,0]-201==index_label:
            #                 label_index=ii
            #         temp_label[index_label].append(temp_data[label_index][:,start:end])
            #
            # for l in range(len(temp_label)):
            #     augmentation_data.append(np.concatenate(temp_label[i]),1)
    processed_data=[]
    for i in range(len(augmentation_data)):
        processed_data.append(np.expand_dims(augmentation_data[i],0))
    return np.concatenate(processed_data,0)

def data_augmentation_by_overlap(data):#对原始数据随机overlap切割后的数据
    expansion_factor=10
    move_step=srate*crop_time-math.ceil((expansion_factor*srate*crop_time-data.shape[1])/(expansion_factor-1))
    overlap_data=[]
    for i in range(expansion_factor):
        temp_data=data[:,i*move_step:i*move_step+math.floor(srate*crop_time)]
        overlap_data.append(np.expand_dims(temp_data,0))
    return np.concatenate(overlap_data,0)

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-3,help='learning rate')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--path',type=str,default='./traindata/')
    parser.add_argument('--subjects',type=list,default=[i+1 for i in range(5)])
    parser.add_argument('--block_num',type=int,default=15)
    opt=parser.parse_args()
    train(opt)