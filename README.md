**Data avaliable**
Please check the Link in 'REST_data' and 'SSVEP_data' folders.
Due to the large dataset size, it is provided via a third-party platform.

**Usage Notes**
The main processing is in 'Preproc_classification.m' script.
Please check the path as the 'path_dataset' and 'save_path' variables in the init param. session.

**SSVEP Data Structure**

    data: chan x time x class x trial
    
    state: Hz
    
    chan_locs: channel name (cell-str)
    
    window: epoch length (ms)
    
    freq: stimuli frequencies (Hz)
    
    phase: stimuli phase
    
    subj: num of subject (str)

**Rest data structure**

    dat_a: chan x time x trial

    resting: pre or post, eye open and eye closed (str)
    
    state: Hz
    
    chan_locs: channel name (cell-str)

**Results**

    Method: classifier (str)
    
    Window len: [a b] (ms)
    
    chan: selected chan (cell-str)

    acc: N-vector (each fold) 

    itr: N-vector (each fold) 

    filter: band pass filter range (Hz)

    fold: N



**Performance Validation Method**

1. **Standard CCA**
    Hakvoort, G., Reuderink, B. & Obbink, M. Comparison of PSDA and CCA detection methods in a SSVEP-based BCI-system. (2011).
2. **Filter bank CCA**
    Chen, X., Wang, Y., Gao, S., Jung, T.-P. & Gao, X. Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain–computer interface. J. Neural Eng. 12, 046008 (2015).
3. **Combined individual template CCA**
    Nwachukwu, S. E. et al. An SSVEP Recognition Method by Combining Individual Template with CCA. Proc. 2019 3rd Int. Conf. Innov. Artif. Intell. 6–10 (2019).
4. **Task-related component analysis-based filter**
  	Nakanishi, M. et al. Enhancing Detection of SSVEPs for a High-Speed Brain Speller Using Task-Related Component Analysis. IEEE Trans. Biomed. Eng. 65, 104–112 (2018)

**Requirements**

1. **EEGLAB toolbox**: used ver. 2022.0
URL: https://sccn.ucsd.edu/eeglab/download.php

Delorme A & Makeig S (2004) EEGLAB: an open-source toolbox for analysis of single-trial EEG dynamics, Journal of Neuroscience Methods 134:9-21.

2. **Fieldtrip toolbox**: used ver. 20190618
URL: https://www.fieldtriptoolbox.org/download/

Oostenveld, R., Fries, P., Maris, E., Schoffelen, JM (2011). FieldTrip: Open Source Software for Advanced Analysis of MEG, EEG, and Invasive Electrophysiological Data. Computational Intelligence and Neuroscience, Volume 2011 (2011)
