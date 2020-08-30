# higgs_diff

Files for ttH-ML differential measurement. The goal is to reconstruct the Higgs momentum spectrum in the 2lSS and 3l ttH-ML channels. This involves (1) identifying the b-jets from the top decay, (2) identifying the decay products of the Higgs, and (3) predicting the Higgs momentum.

Files are organized into folders based on what specific task they are trying to accomplish. Here is an overview of what each of them are for:

truthMatching - match reco jets, leptons, to their corresponding truth particle, and out a root file with the ID of each truth-matched particles parent. Needed for identifying decay products of tops, Higgs

topMatching - Build models for identifying which pair of jets includes the b-jets from the top decays. Useful for identifying Higgs decay products.

higgsMatching - Build models for identifying the decay products of the Higgs

decay3l - In the 3l channel, the Higgs can decay into 2 leptons, or 1 lepton and 2 jets. To reconstruct the Higgs, we need to know which of these to look for. These files try to distringuish between these two decay modes

ptPrediction - Use the output of the matching algorithms, try to predict the pT of the Higgs

sigBkdBDT - Develop models to distinguish between signal (ttH) events and backgrounds

addToRoot - Add the results of the various NNs to a ROOT file to be used in the main analysis

Wmatching - Use similar methods to identify which lepton originated fromt the W decay in ttW events