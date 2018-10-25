#include <incmode>.

#const istop  = "SAT".
#const imin   = 1.
#const imax  = 50.
#const maxgrid = 4.

#program base.

loc(lower_right_ladder).
loc(lower_left_ladder).
loc(key).
loc(right_door).
loc(plat1).
loc(devilleft).


at(plat1,1).
cost(0,1).
cost(0,0).


#program step(k).

{move(L,k)} :- loc(L), at(L1,k),L1!=L.
at(L,k) :- at(L1,k-1),move(L,k-1).


cost(C+50,k) :- move(L,k-1), at(L1,k-1), #count{Z:ro(at(L1),move(L),Z)}=0, not picked(key,k-1), cost(C,k-1).
cost(C+50,k) :- move(L,k-1), at(L1,k-1), #count{Z:ro((at(L1),picked(key)),move(L),Z)}=0, picked(key,k-1), cost(C,k-1).
cost(C+Z,k) :- move(L,k-1), at(L1,k-1), not picked(key,k-1), ro(at(L1),move(L),Z), cost(C,k-1).
cost(C+Z,k) :- move(L,k-1), at(L1,k-1), picked(key,k-1), ro((at(L1),picked(key)),move(L),Z), cost(C,k-1).

picked(key,k) :- at(key,k).

:- picked(key,k),at(right_door,k),move(L,k).
:- at(lower_right_ladder,k),picked(key,k), move(devilleft,k).
:- at(devilleft,k),move(plat1,k).
:- at(plat1,k), not picked(key,k), move(right_door,k).

{at(L,k)} :- at(L,k-1).
cost(Q,k) :- cost(Q,k-1), not move(L,k-1):loc(L).
picked(key,k) :- picked(key,k-1).
:- #count{Q:cost(Q,k)}!=1.
:- #count{L:at(L,k)}!=1.









