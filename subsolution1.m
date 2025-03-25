%% Denoising Subproblem Computation
function subsolution = subsolution1(step,tau,eta,mu,subtolerance)
     step = permute(step,[3 2 1]);
     pars.print = 0;
     pars.tv = 'l1';
     pars.MAXITER = 50;
     pars.epsilon = subtolerance; % Becca used 1e-5;
     subsolution = denoise_bound_3D(step,tau./eta,-mu,1,pars);
     subsolution = permute(subsolution,[3,2,1]);
end           
