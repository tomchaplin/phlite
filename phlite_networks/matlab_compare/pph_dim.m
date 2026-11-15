function pers = pph_dim(A, t, dim)
% Same as pph-matlab's computePPH, but with the homology dimension as an argument
  T = cell(dim+2,1);
  nn = size(A,1);
  for ii = 1:dim+2
    T{ii} = zeros(1, nn*((nn-1)^(ii-1)));
  end
  T_labels = compute_labels(A,T);
  [T_labels, T_allow_times] = compute_times(A,T_labels,t);
  pers = compute_pers(A,T_labels,T_allow_times);
end
