

import sys
import os

from scipy.io.wavfile import read
import numpy as np

from librosa.feature import melspectrogram

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    if len(sys.argv)!=3:
        print("python get_spec_full.py <path_audios> <path_images>")
        sys.exit()


    PATH_AUDIO=sys.argv[1]
    PATH_IMAGE=sys.argv[2]

    NFFT=512
    FRAME_SIZE=0.5
    TIME_SHIFT=0.25
    HOP=64
    NMELS=128
    
    hf=os.listdir(PATH_AUDIO)
    hf.sort()
    print(PATH_AUDIO, len(hf))

    if not os.path.exists(PATH_IMAGE):
        os.makedirs(PATH_IMAGE)
    countbad=0
    countinf=0
    for j in range(len(hf)):
        print("Procesing audio", j, hf[j], len(hf))
        fs, data=read(PATH_AUDIO+hf[j])
        if len(data.shape)>1:
            continue
        data=data-np.mean(data)
        data=data/np.max(np.abs(data))
        file_out=PATH_IMAGE+hf[j].replace(".wav", "")
        if os.path.isfile(file_out):
            continue
        if fs!=16000:
            print("error", fs, j, hf[j])
            sys.exit()
            continue
        
        init=0
        endi=int(FRAME_SIZE*fs)
        nf=int(len(data)/(TIME_SHIFT*fs))-1
        if nf>0:
            mat=np.zeros((1,NMELS,126), dtype=np.float32)
            for k in range(nf):
                try:
                    frame=data[init:endi]
                    imag=melspectrogram(frame, sr=fs, n_fft=NFFT, hop_length=HOP, n_mels=NMELS, fmax=fs/2)
                    init=init+int(TIME_SHIFT*fs)
                    endi=endi+int(TIME_SHIFT*fs)
                    if np.min(np.min(imag))<=0:
                        countinf+=1
                        continue
                    imag=np.log(imag, dtype=np.float32)
                    mat[0,:,:]=imag
                    np.save(file_out+"_"+str(k)+".npy",mat)
                except:
                    init=init+int(TIME_SHIFT*fs)
                    endi=endi+int(TIME_SHIFT*fs)
                    countinf+=1

        else:
            print("WARNING, audio too short", hf[j], len(data))
            countbad+=1
    print(countbad)
    print(countinf)
        



        
        