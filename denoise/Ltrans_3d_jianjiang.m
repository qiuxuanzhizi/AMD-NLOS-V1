function P=Ltrans_3d_jianjiang(X)

[l,m,n]=size(X);

P{1}=X(1:l-1,:,:)-X(2:l,:,:);
P{2}=X(:,1:m-1,:)-X(:,2:m,:);
P{3}=X(:,:,1:n-1)-X(:,:,2:n);
