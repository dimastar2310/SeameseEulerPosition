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

%data 
aint = 4;
%% Convert trianing data into cell arrays
imgCells = mat2cell(imgDat1_validate,127,127,3,ones(aint,1));
vectorCells = mat2cell(imgDat2_validate,255,255,3,ones(aint,1));
imgCells = reshape(imgCells,[aint 1 1]);
vectorCells = reshape(vectorCells,[aint 1 1]);
labelCells = arrayfun(@(x)x,validateLabels_validate.','UniformOutput',false);
combinedCells = [imgCells vectorCells labelCells];
%% Save the converted data so that it can be loaded by filedatastore
save('validationData.mat','combinedCells');
filedatastore = fileDatastore('validationData.mat','ReadFcn',@load);
validationDatastore = transform(filedatastore,@rearrangeData);

%% function to be used to transform the filedatastore 
%to ensure the read(datastore) returns M-by-3 cell array ie., (numInputs+1) columns
function out = rearrangeData(ds)
out = ds.combinedCells;
end