% Face detection and cropped for train image set

img_dir = 'G:\Test\rimset\';
filenames = dir(fullfile(img_dir, '*.jpg'));
num_images = length(filenames);
Fdetector = vision.CascadeObjectDetector;
% Fdetector = vision.CascadeObjectDetector('FrontalFaceLBP');
Fdetector.MergeThreshold = 10;
image_dims =[112, 92 ];

for n = 1:num_images
    filename = fullfile(img_dir, filenames(n).name);
    img = imread(filename);
    img = rgb2gray(img);
    BB = step(Fdetector, img);
    if(size(BB)>=1)
        BB = step(Fdetector, img);
        img= imcrop(img,BB);
%         img= histeq(img);
        img= medfilt2(img);        
        img = imresize(img,image_dims);       
        input_dir= strcat('G:\Test\face_cropped\',filenames(n).name);
        imwrite(img, input_dir);
   end
    
end