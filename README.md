# Deep Learning Methods for Higgs Boson Reconstruction

Files for ttH-ML differential measurement. The goal is to reconstruct the Higgs momentum spectrum in the 2lSS and 3l ttH-ML channels. This involves (1) identifying the b-jets from the top decay, (2) identifying the decay products of the Higgs, and (3) predicting the Higgs momentum.

This repo includes code for completing the following tasks:

[Truth Matching](truthMatching) - match reco jets, leptons, to their corresponding truth particle, and output a root file with the ID of each truth-matched particles parent. Needed for identifying decay products of tops, Higgs

[B-jet Identification](topMatching) - Build models for identifying which pair of jets includes the b-jets from the top decays. Useful for identifying Higgs decay products.

[Higgs Reconstruction](higgsMatching) - Build models for identifying the decay products of the Higgs

[3l Decay Channel](decay3l) - In the 3l channel, the Higgs can decay into 2 leptons, or 1 lepton and 2 jets. To reconstruct the Higgs, we need to know which of these to look for. These files try to distringuish between these two decay modes

[\\(p_T\\) Prediction](ptPrediction) - Use the output of the matching algorithms, try to predict the pT of the Higgs

[Background Rejection](sigBkdBDT) - Develop models to distinguish between signal (ttH) events and backgrounds

[Add Results to ROOT files](addToRoot) - Add the results of the various NNs to a ROOT file to be used in the main analysis

[\\(t\bar{t}W\\) Reconstruction](Wmatching) - Use similar methods to identify which lepton originated fromt the W decay in ttW events