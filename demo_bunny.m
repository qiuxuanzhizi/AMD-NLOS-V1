clear
close all

%% LOAD DATA
obj_list = {'S', 'USAF', 'soap', 'bunny', 'numbers', 'TX', '2019', 'toy'};
obj_name = obj_list{4};
NLOSDATA = loadNLOSDATA(obj_name);

if NLOSDATA.is_confocal
    transient_ = NLOSDATA.transient;
else % extract confocal transient if input is 5D
    transient_ = NLOSDATA.transient_confocal;
end

delta = NLOSDATA.delta;
times = delta:delta:NLOSDATA.times(end);
M = length(times);

% Pad zero
[N, ~, M_] = size(transient_);
if M <= M_
    transient_confocal = transient_;
    M = M_;
    times = linspace(delta, NLOSDATA.times(end), M);
else
    transient_confocal = zeros(N, N, M);
    transient_confocal(:,:,end-M_+1:end) = transient_;
end

%define mask
samp = 64;
mask = definemask(N,samp);


%data = permute(data, [3 2 1]);
data = permute(transient_confocal, [3 2 1]);
[mtx,mtxi] = resamplingOperator(M);
mtx = full(mtx);
mtxi = full(mtxi);

% Set parameters
range = times(end);
width = (max(NLOSDATA.l(:,1)) - min(NLOSDATA.l(:,1)))/2;
%c = (range./M)./delta; 

% Define kernel
psf = definePSF(N, M, width/range);


% numbers of iteration
miniter = 1;
maxiter = 8;

% add gaussian noise
%noise_level = 0.001;
%white_noise = noise_level*randn(size(data));
%data= data + white_noise;

% add impulse noise
%r = 0.001;
%data = imnoise(data,'salt & pepper',r);

% add rice noise
%eta = 0.001;
%data = sqrt((data + eta .* randn(size(data))).^2 + (eta .* randn(size(data))).^2);
 
% Preparing data
grid_z = repmat(times',[1 N N]);
data = data.*(grid_z).^4;
data = permute(data,[3 2 1]);
data = data.* mask;
data = permute(data,[3 2 1]);

% parameters to control TV normalization
tautv = 1;                          
eta = 1e4;

mask = reshape(mask,[1,N,N]);
mask = repmat(mask,[M 1 1]);

% main 
ximage = ATM_solution_2(NLOSDATA, data,mask, psf, tautv, eta, miniter, maxiter);

vol = reshape(mtxi*ximage(:,:),[M N N]);
vol = vol.*(grid_z.^0.5);
vol  = max(real(vol),0);
vol(end-10:end, :, :) = 0;
rho = permute(vol, [3, 2, 1]);

z_min = min(NLOSDATA.z);
z_max = max(NLOSDATA.z);
start_idx = round(((NLOSDATA.target_dist - z_max)*2)  / NLOSDATA.delta);
start_idx = max(start_idx, 1);
end_idx = round(((NLOSDATA.target_dist - z_min)*2) / NLOSDATA.delta);
rho(:,:,end+1:end_idx) = 0;

rho = rho(:,:,start_idx:end_idx);
rho = rho(:,:,end:-1:1); % align with our format (far to near)
rho = rho/max(rho(:));
rho = flip(permute(rho, [3, 2, 1]),3);


tic_z = linspace(0,range./2,size(rho,1));
tic_x = linspace(width,-width,size(rho,2));
tic_y = linspace(width,-width,size(rho,3));

figure('Position', [50, 50, 320, 400]); 
axes('Position', [0.02, 0.212, 0.96, 0.768]); 
imagesc(tic_x,tic_y,squeeze(max(rho,[],1)));
colormap('hot');
axis square;


c = colorbar;
c.Location = 'south'; 
c.Position = [0.02, 0.11, 0.96, 0.08];
c.Limits = [min(rho(:)), max(rho(:))]; 
c.TickLength = 0; 
c.Ticks = [min(rho(:))  + 0.09 * (max(rho(:)) - min(rho(:))), ...
           max(rho(:)) - 0.09 * (max(rho(:)) - min(rho(:)))]; 
c.TickLabels = {num2str(min(rho(:)), '%.2f'), num2str(max(rho(:)), '%.2f')}; 
c.FontName = 'Times New Roman'; 
c.FontSize = 23;
c.AxisLocation = 'out'; 
c.Box = 'on'; 

function psf = definePSF(N, M, slope)
% Local function to compute NLOS blur kernel

x = linspace(-1,1,2*N-1);
y = linspace(-1,1,2*N-1);
z = linspace(0,2,2*M-1);
[grid_z,grid_y,grid_x] = ndgrid(z,y,x);

% Define PSF
psf = abs(((4*slope).^2).*(grid_x.^2 + grid_y.^2) - grid_z);
psf = double(psf == repmat(min(psf,[],1),[2*M-1 1 1]));
psf = psf./norm(psf(:));
psf = circshift(psf,[0 N N]);

end


function [mtx,mtxi] = resamplingOperator(M) % (updated by Byeongjoo)
% Local function that defines resampling operators

mtx = sparse([],[],[],M^2,M,M^2);

x = 1:M^2;
mtx(sub2ind(size(mtx),x,ceil(sqrt(x)))) = 1;
mtx  = spdiags(1./sqrt(x)',0,M^2,M^2)*mtx;

K = kron(speye(M), ones(1, M)); 

mtx = K*mtx;
mtxi = mtx';

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