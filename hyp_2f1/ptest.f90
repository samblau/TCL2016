PROGRAM TEST
  USE TESTING_MODULE
  IMPLICIT NONE
  COMPLEX(KIND=8) :: Z, RES
  Z = 1.28
  RES = TESTFUNC(Z)
  print *, RES
END PROGRAM TEST