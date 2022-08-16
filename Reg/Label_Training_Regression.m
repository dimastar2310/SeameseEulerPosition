
%% Generate random training data for muliple input network
imgDat1_train = uint8(zeros(127, 127, 3, 7)); 
imgDat2_train = uint8(zeros(255, 255, 3, 7)); 
trainLabels_train = zeros(7, 6);



%% labeling for first pair
nike_logo = imread('.\LabelDetectionRegression\Train\1\1.jpg');
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



not_nike = imread('.\LabelDetectionRegression\Train\1\2.jpg');
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

label = [-10
5
-7
0
3.925
1.0471].';

% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 1) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 1) = img2; 
trainLabels_train(1,1:6) = label;



%% labeling for second pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetectionRegression\Train\2\1.jpg');
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



not_nike = imread('.\LabelDetectionRegression\Train\2\2.jpg');
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

label = [1
5
-8
0
4.8
0.875].';
% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 2) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 2) = img2; 
trainLabels_train(2,1:6) = label;



%% labeling for third pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetectionRegression\Train\3\1.jpg');
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



not_nike = imread('.\LabelDetectionRegression\Train\3\2.jpg');
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

label = [-1
2.5
-3
0
4.71
1.5].';
% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 3) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 3) = img2; 
trainLabels_train(3,1:6) = label;



%% labeling for fourth pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetectionRegression\Train\4\1.jpg');
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



not_nike = imread('.\LabelDetectionRegression\Train\4\2.jpg');
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

label = [-0.5
2
-2
0
4.71
1.5].';
% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 4) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 4) = img2; 
trainLabels_train(4,1:6) = label;



%% labeling for fifth pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetectionRegression\Train\5\1.jpg');
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



not_nike = imread('.\LabelDetectionRegression\Train\5\2.jpg');
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

label = [0
2.3
-3
0
4.71
1.57].';
% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 5) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 5) = img2; 
trainLabels_train(5,1:6) = label;



%% labeling for sixth pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetectionRegression\Train\6\1.jpg');
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



not_nike = imread('.\LabelDetectionRegression\Train\6\2.jpg');
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

label = [-0.4
0.5
-0.5
0
4.50
0.52].';
% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 6) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 6) = img2; 
trainLabels_train(6,1:6) = label;

%% labeling for seventh pair
%%%not same pictures folder
nike_logo = imread('.\LabelDetectionRegression\Train\7\1.jpg');
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



not_nike = imread('.\LabelDetectionRegression\Train\7\2.jpg');
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

label = [0
3
-2.5
0
4.71
1.51].';

% Generate random training data for muliple input network
imgDat1_train(1:127, 1:127, 1:3, 7) = img1; 
imgDat2_train(1:255, 1:255, 1:3, 7) = img2; 
trainLabels_train(7,1:6) = label;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


imgDat1_train = double(imgDat1_train)/255.;
imgDat2_train = double(imgDat2_train)/255.;

%% data
aint = 7;
% Convert trianing data into cell arrays
imgCells = mat2cell(imgDat1_train,127,127,3,ones(aint,1));
vectorCells = mat2cell(imgDat2_train,255,255,3,ones(aint,1));

labelCells = mat2cell(trainLabels_train.',6,ones(aint,1));

imgCells = reshape(imgCells,[aint 1 1]);
vectorCells = reshape(vectorCells,[aint 1 1]);
labelCells = reshape(labelCells,[aint 1 1]);
%labelCells = arrayfun(@(x)x,trainLabels_train,'UniformOutput',false);
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
