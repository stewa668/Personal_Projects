program test_random_number
  implicit none
  integer :: i, j
  real :: r(5,5)
  call random_number(r)

  do i=1, 5
    do j=1,5
      write(*,*) r(i,j)
    enddo 
  enddo  

end program
