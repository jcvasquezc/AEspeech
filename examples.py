
# -*- coding: utf-8 -*-
"""
@author: J. C. Vasquez-Correa
        Pattern recognition Lab, University of Erlangen-Nuremberg
        Faculty of Engineering, University of Antioquia,
        juan.vasquez@fau.de
"""

from AEspeech import AEspeech
import os

if __name__=="__main__":

    PATH=os.path.dirname(os.path.abspath(__file__))
    wav_file=PATH+"/audios/pataka.wav"

    aespeech=AEspeech("CAE", 1024) # load the pretrained CAE with 1024 units
    mat_spec=aespeech.compute_spectrograms(wav_file) # compute the decoded spectrograms from the autoencoder
    print(mat_spec.size())
    # aespeech.show_spectrograms(mat_spec)

    bottle=aespeech.compute_bottleneck_features(wav_file)   # compute the bottleneck feaatures from a wav file
    print(bottle.shape)

    error=aespeech.compute_rec_error_features(wav_file) # compute the reconstruction error features from a wav file
    print(error.shape)

    wav_directory=PATH+"/audios/"
    df=aespeech.compute_dynamic_features(wav_directory)   # compute the bottleneck and error-based features from a directory with wav files inside 
                                                          #(dynamic: one feture vector for each 500 ms frame)
    print(df)
    print(df["bottleneck"].shape)
    print(df["error"].shape)
    print(df["wav_file"].shape)
    print(df["frame"].shape)

    df1, df2=aespeech.compute_global_features(wav_directory)  # compute the bottleneck and error-based features from a directory with wav files inside 
                                                              #(static: one feture vector per utterance)
    print(df1)
    print(df2)

    df1.to_csv(PATH+"/bottle_example.csv")
    df2.to_csv(PATH+"/error_example.csv")