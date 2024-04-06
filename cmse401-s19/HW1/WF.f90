!subroutine wfn1d()
program main
implicit none

integer, parameter :: dp=selected_real_kind(15,307)

!real(dp), allocatable :: x#(:)
real(dp) :: x(500), a(500), v(500), y(500)
real(dp) :: t(1000000)
real(dp) :: xmin, xmax, dx
real(dp) :: tmin, tmax, dt
integer :: nx, i, nt, k

xmin = 0.0_dp
xmax = 10.0_dp
nx = size(x)
dx = (xmax - xmin)/dble(nx-1)

tmin = 0.0_dp
tmax = 10.0_dp
nt = 1000000
dt = (tmax - tmin)/dble(nt-1)

x(1) = 0.0_dp

do i = 2, 500
  x(i) = x(i-1) + dx
enddo

t(1) = 0.0_dp

do i = 2, 1000000
  t(i) = t(i-1) + dt
enddo

y = dexp(-(x - 5.0_dp)**2)

!open(20,file='WF.dat')

a(:) = 0.0_dp
v(:) = 0.0_dp

do k=1, nt
  a(2:nx-1) = (y(1:nx-2) + y(3:nx) - 2*y(2:nx-1) )/(dx)**2
  v(:) = v(:) + a(:)*dt
  y(:) = y(:) + v(:)*dt

!  if (mod(k,5000) == 0) then
!    do i=1, 500
!      write(20,*) y(i)
!    enddo
!  endif

enddo

!do i=1, 500
!  write(20,*) y(i)
!enddo


!close(20)

end
