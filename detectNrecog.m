% Face detection and cropped for train image set

img_dir = 'G:\project_work2\FaceSrc\';
filenames = dir(fullfile(img_dir, '*.jpg'));
num_images = length(filenames);
Fdetector = vision.CascadeObjectDetector;
Fdetector.MergeThreshold = 25;
image_dims =[120, 104];
for n = 1:num_images
    filename = fullfile(img_dir, filenames(n).name);
    img = imread(filename);
    img = rgb2gray(img);
    BB = step(Fdetector, img);
    face_img= imcrop(img,BB);     
    img = imresize(face_img,image_dims);
    input_dir= strcat('G:\project_work2\Face_cropped\',filenames(n).name);
    imwrite(img, input_dir);
end

%%

% Input train image directory
input_dir = 'G:\project_work2\Face_cropped\';
image_dims =[120, 104];

filenames = dir(fullfile(input_dir, '*.jpg'));
num_images = length(filenames);
images = [];

% Images converted into a column vector
for n = 1:num_images
    filename = fullfile(input_dir, filenames(n).name);
    img = imread(filename);
    img = im2double(img);
    images(:,n) = img(:);
end

% Average/ mean face calculation
mean_face = mean(images, 2);

% Image difference from the mean image
image_diff =[];

for n=1:num_images
    image_diff(:,n)= images(:,n)-mean_face;

end

% Covarience matrix calculation
image_diff_tr= image_diff';
L = image_diff_tr * image_diff;
% eigen vector and value computation using Principle Component Analysis 
[eig_vec, score, eig_val] = pca(L);
% Large dimension eigen vector
% evec_ui= image_diff *eig_vec;
% we set the no. of k best eigen vector=6
K_best_evec = 6; 
limit=length(eig_val);

for i=1:K_best_evec
   evec_ui(:,i)=image_diff*eig_vec(:,i); 
end

% weight/ the feature vector calculation
% weights = evec_ui' * image_diff;
for i=1:num_images 
    for j=1:K_best_evec
        eig_ui_t=evec_ui(:,j)';
        weights(j,i)=eig_ui_t*image_diff(:,i);
    end
end

% input image for test
[file, path]= uigetfile({'*.jpg'},'Select test image');
fullpath=strcat(path, file);
input_img= imread(fullpath);
input_img = rgb2gray(input_img);
Fdetector = vision.CascadeObjectDetector;
Fdetector.MergeThreshold = 25;
BBox = step(Fdetector, input_img);
input_img= imcrop(input_img, BBox); 
input_img = imresize(input_img,image_dims);

% input image difference and input image weight
input_img = im2double(input_img);
input_img_diff= input_img(:)- mean_face;
input_image_weight= evec_ui' * input_img_diff;

% Euclidian distance between the input image and train images
for n=1:num_images
%     distance(:,n)= 1/(1 + norm( input_image_weight - weights(:,n)));
    distance(:,n)= norm( weights(:,n) - input_image_weight);    
end

% match image score and it's index
[match_score, match_index] = min(distance);

% display the result
figure();
imshow([input_img ,reshape(images(:,match_index), image_dims)]);
colormap(gray);
title(sprintf('matches %s, score %f', filenames(match_index).name, match_score));
%%
% display the eigenvalues
% normalised_evalues = eig_val / sum(eig_val);
% figure, plot(cumsum(normalised_evalues));
% figure, plot(normalised_evalues);
% xlabel('No. of eigenvectors'), ylabel('Variance accounted for');
% xlim([1 60]), ylim([0 1]), grid on;