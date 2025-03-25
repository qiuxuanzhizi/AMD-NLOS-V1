
function v=dxb(u)

v = u(:,:,:) - u([end 1:end-1],:,:);
