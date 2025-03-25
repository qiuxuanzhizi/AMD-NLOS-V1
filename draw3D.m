function draw3D(data, width, thre, ma)
    % thre = 0.2;
    % width = 0.5;
    zdim = size(data, 1);
    xdim = size(data, 2);
    data = flip(flip(data, 2), 3);
    tic_x = linspace(-width, width, xdim);
    tic_y = linspace(-width, width, xdim);
    tic_Z = linspace(0, 96e-4 * zdim / 2, zdim);
    [tic_X, tic_Y] = meshgrid(tic_x, tic_y);
    [I, J] = size(tic_X);
    tic_X = reshape(tic_X, [I * J, 1]);
    tic_Y = reshape(tic_Y, [I * J, 1]);
    [M, dep] = max(data, [], 1);
    M = squeeze(M);
    dep = squeeze(dep);
    dep = tic_Z(dep);
    M = reshape(M, [I * J, 1]);
    dep = reshape(dep, [I * J, 1]);
    M = M / max(M(:));
    M = min(M, ma);
    scatter3(dep(M > thre), tic_X(M > thre), tic_Y(M > thre), M(M > thre) * 50, M(M > thre), 'o', 'filled', 'MarkerFaceAlpha', 1);
    axis([0 tic_Z(end) tic_x(1) tic_x(end) tic_y(1) tic_y(end)])
    colormap('gray')
    caxis([0 max(M(:))])
    view([-64 19])

    set(gca, 'xtick', [], 'xticklabel', [])
    set(gca, 'ytick', [], 'yticklabel', [])
    set(gca, 'ztick', [], 'zticklabel', [])
    % xlabel('z(mm)');
    % ylabel('x(mm)');
    % zlabel('y(mm)');
    axis square;
    set(gca, 'color', 'k');
    set(1, 'defaultfigurecolor', 'w');
    set(gcf, 'position', [700, 350, 300, 300]);

    % 绘制红色边框
    hold on;
    plot3([0, 0, tic_Z(end), tic_Z(end), 0], ...
          [tic_x(1), tic_x(end), tic_x(end), tic_x(1), tic_x(1)], ...
          [tic_y(1), tic_y(1), tic_y(1), tic_y(1), tic_y(1)], 'r-', 'LineWidth', 2);
    plot3([0, 0, tic_Z(end), tic_Z(end), 0], ...
          [tic_x(1), tic_x(end), tic_x(end), tic_x(1), tic_x(1)], ...
          [tic_y(end), tic_y(end), tic_y(end), tic_y(end), tic_y(end)], 'r-', 'LineWidth', 2);
    plot3([0, 0], [tic_x(1), tic_x(1)], [tic_y(1), tic_y(end)], 'r-', 'LineWidth', 2);
    plot3([0, 0], [tic_x(end), tic_x(end)], [tic_y(1), tic_y(end)], 'r-', 'LineWidth', 2);
    plot3([tic_Z(end), tic_Z(end)], [tic_x(1), tic_x(1)], [tic_y(1), tic_y(end)], 'r-', 'LineWidth', 2);
    plot3([tic_Z(end), tic_Z(end)], [tic_x(end), tic_x(end)], [tic_y(1), tic_y(end)], 'r-', 'LineWidth', 2);
    hold off;
end
