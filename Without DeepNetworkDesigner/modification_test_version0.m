%% Generate random training data for muliple input network
clear all 
clc
close all
aint=14;
% Directory
directory = '.\LabelDetection\Train\1\1';
img_dir = fullfile(directory, '1');
depth_dir = fullfile(directory, 'depth');
image_rgb_info  = dir( fullfile(img_dir, '*.jpg'));
image_rgb_filenames = fullfile(img_dir, {image_rgb_info.name} );
image_depth_info = dir( fullfile(depth_dir, '*.png'));
image_depth_filenames = fullfile(depth_dir, {image_depth_info.name} );

%% Generate random training data for muliple input network
imgDat1_train = uint8(zeros(127, 127, 3, 14)); 
imgDat2_train = uint8(zeros(255, 255, 3, 14)); 
trainLabels_train = categorical(zeros(1, 14));



%% labeling for first pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetection\Train\1\1\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\1\1\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(0);

% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 1) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 1) = img2; 
trainLabels_train(1) = label;



%% labeling for second pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetection\Train\1\2\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\1\2\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(0);
% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 2) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 2) = img2; 
trainLabels_train(2) = label;



%% labeling for third pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetection\Train\1\3\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\1\3\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(0);
% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 3) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 3) = img2; 
trainLabels_train(3) = label;



%% labeling for fourth pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetection\Train\1\4\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\1\4\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(0);
% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 4) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 4) = img2; 
trainLabels_train(4) = label;



%% labeling for fifth pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetection\Train\1\5\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\1\5\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(0);
% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 5) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 5) = img2; 
trainLabels_train(5) = label;



%% labeling for sixth pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetection\Train\1\6\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\1\6\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(0);
% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 6) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 6) = img2; 
trainLabels_train(6) = label;

%% labeling for seventh pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetection\Train\1\7\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\1\7\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(0);
% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 7) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 7) = img2; 
trainLabels_train(7) = label;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Correct pairs
%% labeling for eight pair
nike_logo = imread('.\LabelDetection\Train\2\1\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\2\1\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(1);

% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 8) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 8) = img2; 
trainLabels_train(8) = label;



%% labeling for ninth pair
nike_logo = imread('.\LabelDetection\Train\2\2\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\2\2\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(1);

% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 9) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 9) = img2; 
trainLabels_train(9) = label;



%% labeling for tenth pair
nike_logo = imread('.\LabelDetection\Train\2\3\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\2\3\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(1);

% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 10) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 10) = img2; 
trainLabels_train(10) = label;


%% labeling for eleventh pair
nike_logo = imread('.\LabelDetection\Train\2\4\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\2\4\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(1);

% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 11) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 11) = img2; 
trainLabels_train(11) = label;



%% labeling for twelvth pair
nike_logo = imread('.\LabelDetection\Train\2\5\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\2\5\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(1);

% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 12) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 12) = img2; 
trainLabels_train(12) = label;



%% labeling for thirteenth pair
nike_logo = imread('.\LabelDetection\Train\2\6\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\2\6\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(1);

% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 13) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 13) = img2; 
trainLabels_train(13) = label;



%% labeling for fourteenth pair
nike_logo = imread('.\LabelDetection\Train\2\7\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Train\2\7\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(1);

% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 14) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 14) = img2; 
trainLabels_train(14) = label;

imgDat1_train = double(imgDat1_train)/255.;
imgDat2_train = double(imgDat2_train)/255.;
%% Convert trianing data into cell arrays
imgCells = mat2cell(imgDat1_train,127,127,3,ones(aint,1));
vectorCells = mat2cell(imgDat2_train,255,255,3,ones(aint,1));
imgCells = reshape(imgCells,[aint 1 1]);
vectorCells = reshape(vectorCells,[aint 1 1]);
labelCells = arrayfun(@(x)x,trainLabels_train.','UniformOutput',false);
combinedCells = [imgCells vectorCells labelCells];
%% Save the converted data so that it can be loaded by filedatastore
save('traingData.mat','combinedCells');
filedatastore = fileDatastore('traingData.mat','ReadFcn',@load);
trainingDatastore = transform(filedatastore,@rearrangeData);
%% Define muliple input network
layers1 = [
   % imageInputLayer([127 127 3],"Name","data_1")
    imageInputLayer([127 127 3],'Name','input')  
    convolution2dLayer([11 11],96,"Name","Conv1","BiasLearnRateFactor",2,"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu")
    maxPooling2dLayer([3 3],"Name","MaxPooling1","Stride",[2 2])
    groupedConvolution2dLayer([5 5],128,2,"Name","Conv2 Grp","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([3 3],"Name","MaxPooling2","Stride",[2 2])
    convolution2dLayer([3 3],384,"Name","Conv3","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")
    groupedConvolution2dLayer([3 3],192,2,"Name","Conv4 Grp","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_6")
    groupedConvolution2dLayer([3 3],128,2,"Name","conv5 Grp_1","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","batchnorm_8")
    fullyConnectedLayer(10,'Name','fc11')
    additionLayer(2,'Name','add')
    fullyConnectedLayer(2,'Name','fc12')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classOutput')];
lgraph = layerGraph(layers1);
layers2 = [...
    %imageInputLayer([225 225 3],"Name","data_2")
    imageInputLayer([255 255 3],'Name','vinput')
    convolution2dLayer([11 11],96,"Name","Conv1_1","BiasLearnRateFactor",2,"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([3 3],"Name","MaxPooling1_1","Stride",[2 2])
    groupedConvolution2dLayer([5 5],128,2,"Name","Conv2 Grp_1","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    maxPooling2dLayer([3 3],"Name","MaxPooling2_1","Stride",[2 2])
    convolution2dLayer([3 3],384,"Name","Conv3_1","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    groupedConvolution2dLayer([3 3],192,2,"Name","Conv4 Grp_1","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_7")
    groupedConvolution2dLayer([3 3],128,2,"Name","conv5 Grp_2","BiasLearnRateFactor",2)
    batchNormalizationLayer("Name","batchnorm_9")
    fullyConnectedLayer(10,'Name','fc21')];
lgraph = addLayers(lgraph,layers2);
lgraph = connectLayers(lgraph,'fc21','add/in2');
plot(lgraph)
%% Define trainingOptions and also set 'Shuffle' to 'never' for this workaround to work
options = trainingOptions('adam', ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise',...
    'MaxEpochs',7, ...
    'MiniBatchSize',1, ...
    'Verbose',1, ...
    'Plots','training-progress',...
    'Shuffle','never');
net = trainNetwork(trainingDatastore,lgraph,options);

%% VALIDATION

% Directory
directory = '.\LabelDetection\Validate\1\1';
img_dir = fullfile(directory, '1');
depth_dir = fullfile(directory, 'depth');
image_rgb_info  = dir( fullfile(img_dir, '*.jpg'));
image_rgb_filenames = fullfile(img_dir, {image_rgb_info.name} );
image_depth_info = dir( fullfile(depth_dir, '*.png'));
image_depth_filenames = fullfile(depth_dir, {image_depth_info.name} );

%% Generate random validateing data for muliple input network
imgDat1_validate = uint8(zeros(127, 127, 3, 4)); 
imgDat2_validate = uint8(zeros(255, 255, 3, 4)); 
validateLabels_validate = categorical(zeros(1, 4));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% not same pictures 

%% labeling for first pair
nike_logo = imread('.\LabelDetection\Validate\1\1\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Validate\1\1\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(0);

% Generate random validating data for muliple input network
imgDat1_validate(1:127, 1:127, 1:3, 1) = img1; 
imgDat2_validate(1:255, 1:255, 1:3, 1) = img2; 
validateLabels_validate(1) = label;


%% labeling for second pair
nike_logo = imread('.\LabelDetection\Validate\1\2\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Validate\1\2\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(0);

% Generate random validating data for muliple input network
imgDat1_validate(1:127, 1:127, 1:3, 2) = img1; 
imgDat2_validate(1:255, 1:255, 1:3, 2) = img2; 
validateLabels_validate(2) = label;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Correct pairs

%% labeling for third pair
nike_logo = imread('.\LabelDetection\Validate\2\1\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Validate\2\1\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(1);

% Generate random validating data for muliple input network
imgDat1_validate(1:127, 1:127, 1:3, 3) = img1; 
imgDat2_validate(1:255, 1:255, 1:3, 3) = img2; 
validateLabels_validate(3) = label;



%% labeling for third pair
nike_logo = imread('.\LabelDetection\Validate\2\2\1.jpg');
size_logo = size(nike_logo);
if size_logo(1)>size_logo(2)
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[127 NaN]);
img1(1:127,1:size(RGB2,2),1:3) = RGB2;
img1 = uint8(img1);
else
img1 = zeros(127,127);
RGB2 = imresize(nike_logo,[NaN 127]);
img1(1:size(RGB2,1),1:127,1:3) = RGB2;
img1 = uint8(img1);
end



not_nike = imread('.\LabelDetection\Validate\2\2\2.jpg');
size_mall = size(not_nike);

if size_mall(1)>size_mall(2)
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[255 NaN]);
img2(1:255,1:size(RGB2,2),1:3) = RGB2;
img2 = uint8(img2);
else
img2 = zeros(255,255);
RGB2 = imresize(not_nike,[NaN 255]);
img2(1:size(RGB2,1),1:255,1:3) = RGB2;
img2 = uint8(img2);
end

label = categorical(1);

% Generate random validating data for muliple input network
imgDat1_validate(1:127, 1:127, 1:3, 4) = img1; 
imgDat2_validate(1:255, 1:255, 1:3, 4) = img2; 
validateLabels_validate(4) = label;

imgDat1_validate = double(imgDat1_validate)/255.;
imgDat2_validate = double(imgDat2_validate)/255.;

aint = 4;
%% Convert trianing data into cell arrays
imgCells = mat2cell(imgDat1_validate,127,127,3,ones(aint,1));
vectorCells = mat2cell(imgDat2_validate,255,255,3,ones(aint,1));
imgCells = reshape(imgCells,[aint 1 1]);
vectorCells = reshape(vectorCells,[aint 1 1]);
labelCells = arrayfun(@(x)x,validateLabels_validate.','UniformOutput',false);
combinedCells = [imgCells vectorCells];
%% Save the converted data so that it can be loaded by filedatastore
save('validationData.mat','combinedCells');
filedatastore = fileDatastore('validationData.mat','ReadFcn',@load);
trainingDatastore = transform(filedatastore,@rearrangeData);



YTest = classify(net,trainingDatastore);

figure
confusionchart(validateLabels_validate,YTest)

accuracy = mean(YTest == validateLabels_validate);

idx = randperm(size(imgDat1_validate,4),aint);
figure
tiledlayout(3,3)
for i = 1:aint
    nexttile
    I = [imgDat1_validate(:,:,:,idx(i)) imresize(imgDat2_validate(:,:,:,idx(i)),[127 NaN])];
    imshow(I)

    label = string(YTest(idx(i)));
    title("Predicted Label: " + label)
end
%% function to be used to transform the filedatastore 
%to ensure the read(datastore) returns M-by-3 cell array ie., (numInputs+1) columns
function out = rearrangeData(ds)
out = ds.combinedCells;
end