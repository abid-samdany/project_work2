% read in tiff image and convert it to double format
my_image = im2double(imread('trail.PNG'));
my_image = my_image(:,:,1);
%disp(my_image);

% allocate space for thresholded image
image_thresholded = zeros(size(my_image));
% loop over all rows and columns
for ii=1:size(my_image,1)
    for jj=1:size(my_image,2)
        % get pixel value
        pixel=my_image(ii,jj);
        %disp(pixel);
          % check pixel value and assign new value
          
    end
    
end
% display result
figure()
subplot(1,2,1)
imshow(my_image,[])
title('original image')
subplot(1,2,2)
imshow(image_thresholded,[])
title('thresholded image')
