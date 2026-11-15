% Reads networks from infile, writes bars to outfile.
% infile format: blocks "n dim t_end" then n rows of the weight matrix
function run_matlab(infile, outfile)
  addpath(getenv('PPH_MATLAB_DIR'));
  fin = fopen(infile, 'r'); fout = fopen(outfile, 'w');
  k = 0;
  while true
    hdr = fscanf(fin, '%f', 3);
    if numel(hdr) < 3, break; end
    n = hdr(1); dim = hdr(2); tend = hdr(3);
    A = fscanf(fin, '%f', [n n])';
    pers = pph_dim(A, tend, dim);
    fprintf(fout, 'network %d\n', k);
    for d = 1:dim+1
      b = pers{d};
      for r = 1:size(b,1)
        fprintf(fout, '%d %.17g %.17g\n', d-1, b(r,1), b(r,2));
      end
    end
    k = k + 1;
  end
  fclose(fin); fclose(fout);
end
