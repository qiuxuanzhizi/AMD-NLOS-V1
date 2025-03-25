
function v=dzb(u)

v = u(:,:,:) - u(:,:,[end 1:end-1]);
