#program check(k).
:- query(k), cost(C,k), C <= -100.
:- query(k), cost(0,k).