function ximage = ATM_solution_2(NLOSDATA, y, mask, psf, tau, eta, miniter, maxiter)
SNR = 1e-1;
y = gpuArray(single(y));
[bin, N, ~] = size(y);
[mtx,mtxi] = resamplingOperator(bin);
mtx = full(mtx);
mtxi = full(mtxi);
mtx = gpuArray(single(full(mtx)));
mtxi = gpuArray(single(full(mtxi)));
mask = gpuArray(single(mask));

% utility functions
square2cube = @(x) reshape(x, [], N, N);
cube2square = @(x) x(:,:);
Ffun = @(x)  fftn(x);
Ftfun = @(x) ifftn(x);
pad_array = @(x) fill_data(x,bin,N);
trim_array1 = @(x) select_data(x,bin,N);
psf = gpuArray(single(psf));
fpsf = gpuArray(single(fftn(psf)));
invpsf = conj(fpsf) ./ (abs(fpsf).^2 + 1./SNR);
A = @(x) real ((square2cube(mtxi*cube2square(trim_array1(real(Ftfun(fpsf .*Ffun(pad_array(x))))))))).*mask;
AT = @(x) trim_array1(real(Ftfun(invpsf .* Ffun(pad_array((square2cube(mtx*cube2square(x))))))));

%% Initialization
%u = AT(y);
u = zeros(size(y));
u = gpuArray(single(zeros(size(y))));
alpha = [1/2 1/2];
alpha = gpuArray(single(alpha));
sigma2 = [0.005 10];
sigma2 = gpuArray(single(sigma2));

%omega2 = zeros(size(y));
omega = gpuArray(single(zeros([size(y), 2])));
omega2 = gpuArray(single(zeros(size(y)))); 
omega2(y>=.95) = 1;
omega2(y<=0.05) = 1;
omega1 = 1-omega2;
omega(:,:,:,1) = omega1;
omega(:,:,:,2) = omega2;

w = omega(:,:,:,1)/(sigma2(1)+eps) + omega(:,:,:,2)/(sigma2(2)+eps);

iter = 1;
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
    sigma2(1) = max(sigma2(1), 1e-6);
    sigma2(2) = max(sigma2(2), 1e-6);
    w=omega(:,:,:,1)/(sigma2(1)+eps)+omega(:,:,:,2)/(sigma2(2)+eps);
    iter = iter + 1;
    
    vol  = reshape(mtxi*f(:,:),[bin N N]);
    vol  = max(real(vol),0);
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
    rho = flip(flip(permute(rho, [3, 2, 1]),1),2);
    figure(1);draw3D(gather(rho),0.5,0.2,1);drawnow
end
ximage = gather(f);

function [mtx,mtxi] = resamplingOperator(M) % (updated by Byeongjoo)
% Local function that defines resampling operators

mtx = sparse([],[],[],M^2,M,M^2);

x = 1:M^2;
mtx(sub2ind(size(mtx),x,ceil(sqrt(x)))) = 1;
mtx  = spdiags(1./sqrt(x)',0,M^2,M^2)*mtx;

K = kron(speye(M), ones(1, M)); 

mtx = K*mtx;
mtxi = mtx';

function tdata = fill_data(data,M,N)
tdata = zeros(2*M-1,2*N-1,2*N-1);
tdata(1:M,1:N,1:N) = data;

function data = select_data(tdata,M,N)
data = tdata(1:M,1:N,1:N);
