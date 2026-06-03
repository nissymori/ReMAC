# distance-only M=1, 2, 4, 8
python quad_opt_update_norm.py --m 1 --distance-only --add-y-label
python quad_opt_update_norm.py --m 2 --distance-only --no-legend
python quad_opt_update_norm.py --m 4 --distance-only --no-legend
python quad_opt_update_norm.py --m 8 --distance-only 

# opt-path-only M=1, 2, 4, 8
python quad_opt_update_norm.py --m 1 --optimize-only --add-y-label
python quad_opt_update_norm.py --m 2 --optimize-only --no-legend
python quad_opt_update_norm.py --m 4 --optimize-only --no-legend
python quad_opt_update_norm.py --m 8 --optimize-only 