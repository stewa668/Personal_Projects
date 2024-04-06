!subroutine transpose()
program main
implicit none

integer, parameter :: n = 20000, bsize = 16
real , allocatable :: matI(:,:), matO(:,:)
integer :: i, j, k, l

allocate(matI(n,n), matO(n,n))
!matI(:,:) = 0.0
matO(:,:) = 0.0

call random_number(matI)

do i = 1, n, bsize
  do j = 1, n, bsize
    do k = i, min(i+bsize-1,n)
      do l = j, min(j+bsize-1,n)
        matO(l,k) = matI(k,l)
      enddo
    enddo
  enddo
enddo

deallocate(matO, matI)

end
