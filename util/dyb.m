
function v=dyb(u)

v = u(:,:,:) - u(:,[end 1:end-1],:);
