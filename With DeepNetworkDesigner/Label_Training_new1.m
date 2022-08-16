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

%% data
aint = 14;
% Convert trianing data into cell arrays
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

%% function to be used to transform the filedatastore 
%to ensure the read(datastore) returns M-by-3 cell array ie., (numInputs+1) columns
function out = rearrangeData(ds)
out = ds.combinedCells;
end
