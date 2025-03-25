close all
clear
load('./data_new/dragon/meas_30min.mat');
load('./data_new/dragon/tof.mat'); 
target_size1 = [128,128,2048];
measlr = imresize3d(meas, 128, 128, 2048);
target_size2 = [128,128];
tofgridlr = imresize(tofgrid,target_size2);

wall_size = 2;
width = wall_size/2;
bin_resolution = 32e-12;
c = 3*10^8;

for ii = 1:size(measlr,1)
    for jj = 1:size(measlr,2)
        measlr(ii, jj, :) = circshift(measlr(ii, jj, :), [0, 0, -floor(tofgridlr(ii, jj) / (bin_resolution*1e12))]);
    end
end

%measlr = (measlr(1:2:end,:,:) + measlr(2:2:end,:,:));
%measlr = (measlr(:,1:2:end,:) + measlr(:,2:2:end,:));

crop = 512;
measlr = measlr(:, :, 1:crop);

bin = size(measlr,3); % Spatial resolution of data
N = size(measlr,1);   % Temporal resolution of data
range = bin.*c.*bin_resolution;

% Define transform operators
[mtx,mtxi] = resamplingOperator(bin);
mtx = full(mtx);
mtxi = full(mtxi);

% Define mask
samp = 128;                           % scan points samp x samp
mask = definemask(N,samp);

% Define PSF
psf = definePsf(N,width,bin,bin_resolution,c);

% Preparing data
y = measlr.*mask;
y = permute(y,[3 2 1]);

disp(size(y))

% numbers of iteration
miniter = 1;
maxiter = 8;

grid_z = repmat(linspace(0,1,bin)',[1 N N]);
y = y.*(grid_z.^4);

% parameters to control TV normalization
tautv = 1;                          
eta = 1e3;

% main 
ximage = ATM_solution_1(mask, y, psf, samp, tautv, eta, miniter, maxiter);

vol = reshape(mtxi*ximage(:,:),[bin N N]);
vol = vol.*(grid_z.^0.5);
vol  = max(real(vol),0);
vol(end-50:end, :, :) = 0;
vol = permute(vol, [1, 3, 2]);
result = permute(vol, [2, 3, 1]);
vol = flip(vol, 2);
vol = flip(vol, 3);


tic_z = linspace(0,range./2,size(vol,1));
tic_y = linspace(width,-width,size(vol,2));
tic_x = linspace(width,-width,size(vol,3));

figure
imagesc(tic_x,tic_y,squeeze(max(vol,[],1)));
colormap('gray');
axis square;
axis off;


function [mtx,mtxi] = resamplingOperator(M)   % refer to lct reconstruction
 % Local function that defines resampling operators
     mtx = sparse([],[],[],M.^2,M,M.^2);
     
     x = 1:M.^2;
     mtx(sub2ind(size(mtx),x,ceil(sqrt(x)))) = 1;
     mtx  = spdiags(1./sqrt(x)',0,M.^2,M.^2)*mtx;
     mtxi = mtx';
     
     K = log(M)./log(2);
     for k = 1:round(K)
          mtx  = 0.5.*(mtx(1:2:end,:)  + mtx(2:2:end,:));
          mtxi = 0.5.*(mtxi(:,1:2:end) + mtxi(:,2:2:end));
     end

end


function psf = definePsf(N,width,bin,timeRes,c)     % refer to lct reconstruction
    linexy = linspace(-2*width,2*width,2*N-1);            % linspace of scan range
%   linexy = linspace(-2*width,2*width,2*N);    
    range = (bin*timeRes*c/2)^2;                    % total range of t^2 domain: bin^2*timeRes^2
    gridrange = bin*(timeRes*c/2)^2;                % one grid in t^2 domain: bin*timeRes^2
    [xx,yy,squarez] = meshgrid(linexy,linexy,0:gridrange:range);
    blur = abs((xx).^2+(yy).^2-squarez+0.0000001);
    blur = double(blur == repmat(min(blur,[],3),[ 1 1 bin+1 ]));
    blur = blur(:,:,1:bin);                               % generate light-cone
    psf = zeros(2*N-1,2*N-1,2*bin); 
%     psf = zeros(2*N,2*N,2*bin); 
%     psf(2:2*N,2:2*N,bin+1:2*bin) = blur;
    psf(:,:,bin+1:2*bin) = blur;                          % place it at center
end


function mask = definemask(N,samp)
    xx = linspace(1,N,samp);
    yy = linspace(1,N,samp);
    mask = zeros(N,N); 
    for i = 1:samp
        for j = 1:samp
            mask(round(xx),round(yy)) = 1;
        end
    end
end