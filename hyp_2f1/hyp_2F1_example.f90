PROGRAM HYP_TEST
  USE HYP_2F1_MODULE
  IMPLICIT NONE
  REAL(KIND=8)    :: TEST,TEST_2F1
  COMPLEX(KIND=8) :: A,B,C,Z,F,HYP_2F1

  READ(*,*) A,B,C,Z

  OPEN(UNIT=10,FILE='hyp_2F1_test.output')

  WRITE(UNIT=10,FMT=*) 'a:',A
  WRITE(UNIT=10,FMT=*) 'b:',B
  WRITE(UNIT=10,FMT=*) 'c:',C
  WRITE(UNIT=10,FMT=*) 'z:',Z
  WRITE(UNIT=10,FMT=*) 

  !print *, 'Before the function'
  !print *, A
  !print *, B
  !print *, C
  !print *, Z

  
  F = HYP_2F1(A,B,C,Z)
  TEST = TEST_2F1(A,B,C,Z,F)
  
  WRITE(UNIT=10,FMT=*) 'F:',F
  WRITE(UNIT=10,FMT=*) 'test:',TEST
  CLOSE(UNIT=10)
END PROGRAM HYP_TEST
