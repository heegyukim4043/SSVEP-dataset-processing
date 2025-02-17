**Data structure**
    data: chan x time x class x trial
    state: Hz
    chan_locs: channel name (cell-str)
    window: epoch length (ms)
    freq: stimuli frequencies (Hz)
    phase: stimuli phase
    subj: num of subject (str)


Method 
1. **Standard CCA**
    Hakvoort, G., Reuderink, B. & Obbink, M. Comparison of PSDA and CCA detection methods in a SSVEP-based BCI-system. (2011).
2. **Filter bank CCA**
    Chen, X., Wang, Y., Gao, S., Jung, T.-P. & Gao, X. Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain–computer interface. J. Neural Eng. 12, 046008 (2015).
3. **Combined individual template CCA**
    Nwachukwu, S. E. et al. An SSVEP Recognition Method by Combining Individual Template with CCA. Proc. 2019 3rd Int. Conf. Innov. Artif. Intell. 6–10 (2019).
4. **Task-related component analysis-based filter**
  	Nakanishi, M. et al. Enhancing Detection of SSVEPs for a High-Speed Brain Speller Using Task-Related Component Analysis. IEEE Trans. Biomed. Eng. 65, 104–112 (2018)

**Matlab toolbox **

1. EEGLAB toolbox: ver. 2022.0
URL: https://sccn.ucsd.edu/eeglab/download.php

2. Fieldtrip toolbox: ver. 20190618
URL: https://www.fieldtriptoolbox.org/download/
