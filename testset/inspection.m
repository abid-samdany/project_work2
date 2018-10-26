% Input train image directory
input_dir = 'G:\Test\ATT\';
image_dims =[120, 104];

filenames = dir(fullfile(input_dir, '*.png'));
num_images = length(filenames);
images = [];

% Images converted into a column vector
for n = 1:num_images
    filename = fullfile(input_dir, filenames(n).name);
    img = imread(filename);
%     img = rgb2gray(img); 
    img = im2double(img);
    img = imresize(img,image_dims);
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
evec_ui= image_diff *eig_vec;
% we set the no. of eigenface=10
num_eigenfaces = 15;
evec_ui = evec_ui(:, 1:num_eigenfaces);

% weight/ the feature vector calculation
weights = evec_ui' * image_diff;

%%

% input image for test
input_img= imread('10_3.png');
% input_img = rgb2gray(input_img); 
input_img = imresize(input_img,image_dims);

% input image difference and input image weight
input_img = im2double(input_img);
input_img_diff= input_img(:)- mean_face;
input_image_weight= evec_ui' * input_img_diff;

% Euclidian distance between the input image and train images
for n=1:num_images
    distance(:,n)= 1 / (1+ norm( weights(:,n) - input_image_weight));
end

% match image score and it's index
[match_score, match_index] = max(distance);

% display the result
figure();
imshow([input_img ,reshape(images(:,match_index), image_dims)]);

colormap(gray);
title(sprintf('matches %s, score %f', filenames(match_index).name, match_score));

%%
% % display the eigenvectors
% figure;
% for n = 1:num_eigenfaces
% subplot(2, ceil(num_eigenfaces/2), n);
% evect = reshape(evec(:,n), image_dims);
% imagesc(evect);
% colormap(gray); 
% end
% 
% % display the eigenvalues
% normalised_evalues = evalues / sum(evalues);
% figure, plot(cumsum(normalised_evalues));
% xlabel('No. of eigenvectors'), ylabel('Variance accounted for');
% xlim([1 30]), ylim([0 1]), grid on;