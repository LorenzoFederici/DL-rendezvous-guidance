filename1 = indir.'Trajectory.txt'
filename2 = 'missions/MSR.dat'
filename3 = 'missions/CVX_sol.dat'

set term unknown

set key top right box opaque height 0.5
set grid 
set xrange [-18:0]
set yrange [-6:6]
set size ratio -1

set xlabel "$y$"
set ylabel "$x$"

set format x "%.1f"
set format y "%.1f"

set autoscale fix
set xrange [] reverse

set lt 10 dt '--' lw 3.5 lc rgb "red"
set lt 11 lw 1.5 lc rgb "blue"
set ls 2 pt 4 ps 2 lc rgb "#228B22"

set arrow 1 nohead from first 0, first 0 length first 16 angle 160 lw 2 lc rgb "black"
set arrow 2 nohead from first 0, first 0 length first 16 angle 200 lw 2 lc rgb "black"

plot filename3 using 3:2 w l lt 10 title "convex"
replot filename1 using 2:1 w l lt 11 title "DNN"
replot filename2 every ::::0 using 3:2 w p ls 2 notitle
replot filename2 every ::1 using 3:2 w p ls 2 notitle

set terminal epslatex standalone color colortext 10 lw 2 header \
"\\usepackage{amsmath}\n\\usepackage{siunitx}"
set output 'traj2D_gnuplot.tex'
replot