function u = subsolution2(A, AT, eta,b,w,u)
%CGsolver for Ax=b;
x = u;
gd = Ax(A,AT,eta,w,x)-b;
p  = -gd;
delta_old = sum(sum(sum(gd.^2)));
xold = x;
iter = 0;
subminiter = 1;
submaxiter = 40;
while (iter <= subminiter) || (iter <= submaxiter)
    h = Ax(A, AT, eta, w, p);
    tau = delta_old/(sum(sum(sum(p.*h)))+eps);
    x = x + tau*p;
    gd = gd + tau*h;
    delta = sum(sum(sum(gd.^2)));
    belta = delta/(delta_old + eps);
    p = -gd + belta*p;
    delta_old = delta;
    error = max(max(max(abs(xold-x))));
    fprintf('error is %f \n',error);
    xold = x;
    iter = iter+1;
    if error<5e-4
        u = x;
        break;
    end
end

function Axvalue=Ax(A,AT,eta,w,x)

a = A(x);
b = AT(w.*a);
Axvalue = b + eta*x;
