%% Before first time use set MATLAB path with subfolders and run vl_setupnn
layer = MyCrossCorr(2,'crossCorrilation');
validInputSize = {[6 6 128],[22 22 128]};
checkLayer(layer,validInputSize,'ObservationDimension',4)