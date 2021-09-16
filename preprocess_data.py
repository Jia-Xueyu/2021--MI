import pickle
import numpy as np
from sklearn import preprocessing

class MIData:
    def __init__(self,path,subjects,block_num):
        self.path=path
        self.subjects=subjects
        self.block_num=block_num

    def crop_data(self):
        data=[]
        trigger_ch=64
        trial_start=240
        trial_end=241
        trial_start=np.float(trial_start)
        trail_end=np.float(trial_end)
        for subject in self.subjects:
            path = self.path +'S0'+ str(subject) + '/'
            block_data_path = ['block_' + str(i + 1) + '.pkl' for i in range(self.block_num)]

            for i in range(self.block_num):
                temp_path = path + block_data_path[i]
                f = open(temp_path, 'rb')
                temp_data = pickle.load(f)
                eeg_data=temp_data['data']
                trigger_index=eeg_data[trigger_ch,:]
                start_index=np.where(trigger_index==trial_start)
                end_index=np.where(trigger_index==trial_end)
                start_index,end_index=list(start_index)[0].tolist(),list(end_index)[0].tolist()
                assert len(start_index)==len(end_index), 'the nums of start and end are different.'
                for j in range(len(start_index)):
                    temp_eeg_data=eeg_data[:,start_index[j]+1:end_index[j]]
                    temp_eeg_data[0:64,:]=preprocessing.scale(temp_eeg_data[0:64,:], axis=1)
                    temp_eeg_data[64,:]=temp_eeg_data[64,:]-201
                    data.append({'subject':subject,'block':i,'data':temp_eeg_data})

        return data



if __name__=='__main__':
    path='./traindata/'
    subjects=[i+1 for i in range(1)]
    block_num=15
    MIData=MIData(path,subjects,block_num)
    data=MIData.crop_data()
    print(data.shape)