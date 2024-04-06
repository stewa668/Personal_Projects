!subroutine transpose()
program main
implicit none

integer, parameter :: n = 20000
real , allocatable :: matI(:,:), matO(:,:)
integer :: i, j

allocate(matI(n,n), matO(n,n))

call random_number(matI)

do i = 1, n
  do j = 1, n
    matO(j,i) = matI(i,j)
  enddo
enddo

deallocate(matO, matI)

end
