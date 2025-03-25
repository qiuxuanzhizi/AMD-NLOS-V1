function ximage = ATM_solution(mask, y, psf, jit, samp, tau, eta, miniter, maxiter)

y = gpuArray(single(y));
[bin, N, N] = size(y);
bin_resolution = 32e-12;            % Time resolution
wall_size = 1; % optional
width = wall_size/2;
c = 3*10^8;        % speed of light
range = bin.*c.*bin_resolution;  %
slope = width./range;
[mtx,mtxi] = resamplingOperator(bin);
%mtx = full(mtx);
%mtxi = full(mtxi);
mtx = gpuArray(single(full(mtx)));
mtxi = gpuArray(single(full(mtxi)));
grid_z = repmat(linspace(0,1,bin)',[1 N N]);

mask = definemask(N,samp);
mask = reshape(mask,[1,N,N]);
mask = repmat(mask,[bin 1 1]);
mask = single(mask);
mask = gpuArray(mask);

% utility functions
square2cube = @(x) reshape(x, [],N,N);
cube2square = @(x) x(:,:);
Ffun = @(x)  fftn(x);
Ftfun = @(x) ifftn(x);

pad_array1 = @(x) padarray(x, [bin/2, N/2-1, N/2-1],'pre');
pad_array2 = @(x) padarray(x, [bin/2, N/2+1, N/2+1],'post');
trim_array1 = @(x) x(bin/2+1:3*bin/2, N/2:3*N/2-1, N/2:3*N/2-1);
trim_arrayjit = @(x) x(4:3+bin,:,:);
psf = permute(psf,[3 2 1]);
psf = gpuArray(single(psf));
psf1 = padarray(psf,[0,1,1],'pre');
psfFT = fftn(psf1);
psfFT1 = fftn(padarray(flip(flip(flip(psf,1),2),3),[0,1,1],'pre'));
jit = permute(jit,[3 2 1]);
jit = gpuArray(single(jit));
jitFT = fftn(padarray(jit,[bin-1,N-1,N-1],'post'));
    
A = @(x) real(trim_arrayjit(Ftfun(jitFT.*Ffun(padarray(square2cube(mtxi*cube2square(trim_array1(real(ifftshift(Ftfun(psfFT .* Ffun(pad_array2(pad_array1(x))))))))),[6 0 0],'post'))))).*mask;  
AT = @(x) trim_array1(real(ifftshift(Ftfun(psfFT1 .* Ffun(pad_array2(pad_array1(square2cube(mtx*cube2square(x)))))))));


%% Initialization
u = gpuArray(single(zeros(size(y))));
alpha = [1/2 1/2];
alpha = gpuArray(single(alpha));
sigma2 = [0.005 10];
sigma2 = gpuArray(single(sigma2));

omega = gpuArray(single(zeros([size(y), 2])));
omega2 = gpuArray(single(zeros(size(y)))); 
omega2(y>=.95) = 1;
omega2(y<=0.05) = 1;
omega1 = 1-omega2;
omega(:,:,:,1) = omega1;
omega(:,:,:,2) = omega2;

w = omega(:,:,:,1)/(sigma2(1)+eps) + omega(:,:,:,2)/(sigma2(2)+eps);

iter = 0;
mu = 0;
subtolerance = 1e-5;

%% Begin Main Algorithm Loop
while (iter <= miniter) || (iter <= maxiter)
    disp(iter)
    %sub-problem 1
    f = subsolution1(u, tau, eta, mu, subtolerance);
    f(f<0) = 0;
    f(f>1) = 1;

    %sub-problem 2
    b = AT(w.*y) + eta*f;
    u = subsolution2(A,AT,eta,b,w,u);
    u(u<0)=0;
    u(u>1)=1;

    %sub-problem 3
    d = A(u) - y;
    [alpha, sigma2, omega] = subsolution3(d, alpha, sigma2);
    w=omega(:,:,:,1)/(sigma2(1)+eps)+omega(:,:,:,2)/(sigma2(2)+eps);
    iter = iter + 1;
    vol(:,:,:) = abs(f(:,:,:));
    vol(end-50:end, :, :) = 0;

    result = reshape(mtxi*f(:,:),[bin N N]);
    ximage_result = result.*(grid_z.^0.5);    % convert signal to t domain
    
    figure(1);draw3D(gather(vol),0.5,0.2,1);drawnow
end

ximage = f;

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


 function mask = definemask(N,samp)
    xx = linspace(1,N,samp);
    yy = linspace(1,N,samp);
    mask = zeros(N,N); 
    for i = 1:samp
        for j = 1:samp
            mask(round(xx),round(yy)) = 1;
        end
    end

 function [ps, ss] = psnr1(vol)
    % 根据 vol 的数据类型选择 xref 的数据类型
    if isa(vol, 'single')
        xref_type = 'single';
    elseif isa(vol, 'double')
        xref_type = 'double';
    else
        error('Unsupported data type for vol. Expected single or double.');
    end

    % 处理输入数据
    x = vol;
    x = x / max(x(:)); % 归一化
    x1 = squeeze(max(abs(x), [], 1)); % 取最大值并压缩维度

    % 加载参考数据并根据类型转换
    load('bowlingscene.mat');
    xref = cast(scene, xref_type); % 根据 vol 的类型转换 xref
    xref = xref / max(xref(:)); % 归一化
    xref1 = squeeze(max(abs(xref), [], 1)); % 取最大值并压缩维度

    % 计算 PSNR 和 SSIM
    ps = psnr(x1, xref1);
    ss = ssim(x1, xref1);

