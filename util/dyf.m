function v=dyf(u)

v = u(:,[2:end 1],:) - u(:,:,:);
