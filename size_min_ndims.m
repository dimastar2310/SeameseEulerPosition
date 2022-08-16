function sz = size_min_ndims(x, m)
% Ensure that number of dimensions is at least m.
n = max(ndims(x), m);
sz = size_ndims(x, n);
end
