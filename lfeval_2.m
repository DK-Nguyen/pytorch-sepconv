% ICME 2018 Grand Challenge on Densely Sampled Light Field Reconstruction
% http://www.tut.fi/civit/index.php/icme-2018-grand-challenge-densely-sampled-light-field-reconstruction/
% 
% Light field evaluation tool
%
% For each of the three challenge  categories, reconstructed 193 views are 
% compared against the GT in terms of per-view PSNR obtained as a log ratio 
% between 255^2 and the average of the per-view MSEs of the R, G and B color channels. 
% The lowest per-view PSNR for each dataset will be selected as the single quality measure for the given dataset. 
%
% recLFpath - path to reconstructed light field 
% groundLFPath - path to ground truth light field
% 
% err - evaluation error [GrandChallenge Gao]
% errPerView - error per view

function [ err, errPerView ] = lfeval_2(recLFpath, groundLFPath)
    
    NumOfImages = 193;
    errPerView = zeros(NumOfImages, 1);
    
    parfor i = 1:NumOfImages
        recfn = fullfile(recLFpath, sprintf('%04i.png', i));
        gtfn = fullfile(groundLFPath, sprintf('%04i.png', i));
        
        recimg = imread(recfn);
        gtimg = imread(gtfn);
        
        recimg = double(recimg);
        gtimg  = double(gtimg);
        e = abs(recimg - gtimg).^2;
        mse  = sum(e(:))/numel(e);
        psnr_err = 10*log10(255*255/mse);
        
%         errPerView(i,1) = psnr_err;
        
        rec_yc = rgb2ycbcr(recimg);
        rec_gt = rgb2ycbcr(gtimg);
        errPerView(i,:) = [psnr_err psnr(rec_yc(:,:,1), rec_gt(:,:,1), 255)];       
        
    end
    
    err(1) = min(errPerView(1:end,1));
    err(2) = mean(errPerView(1:end,2));
    
    