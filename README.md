# AEspeech

Feature extraction from speech signals using pre-trained autoencoders

This repository contains the architecture and trained models for the paper "Parallel Representation Learning for the Classification of Pathological Speech: studies on Parkinson's Disease and Cleft Lip and Palate", submitted Speech Communications, 2020

## Feature extraction from speech signals using the pre-trained models

 See [`examples`](examples.py) for usage.

```python
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
```

## Train new models

In order to train new autoencoders with additional data, please check

1. [`get_spec_full`](get_spec_full.py) to generate the input spectrogras to train the models.

2. [`TrainCAE`](TrainCAE.py) or  [`TrainRAE`](TrainRAE.py) to train a convolutional or recurrent autoencoder, respectively.


## Reference
If AEspeech is used for your own research please cite the following publication: Parallel Representation Learning for the Classification of Pathological Speech: Studies on Parkinson's Disease and Cleft Lip and Palate
```
@article{vasquez2020parallel,
author = {Vasquez-Correa, J. C. and Arias-Vergara T. and Schuster, M. and Orozco-Arroyave, J. R. and Nöth, E.},
year = {2020},
month = {},
pages = {},
title = {Parallel Representation Learning for the Classification of Pathological Speech: Studies on Parkinson's Disease and Cleft Lip and Palate},
volume = {},
journal = {Speech communication (Under review)},
doi = {}
}
```
## License
MIT License Copyright (c) 2020 J. C. Vasquez-Correa


