      program wave
      implicit real*8(a-h,o-z),integer*4(i-n)
      parameter (nx=500,nt=100000,nd=50)
      dimension x(nx)
      dimension y(nx)
      dimension v(nx)
      dimension ylap(nx)
      dimension t(nt)

      xmin = 0.0
      xmax = 10.0
      call linspace(nx,xmin,xmax,x)
      tmin = 0.0
      tmax = 10.0
      call linspace(nt,tmin,tmax,t)
      do i=1,nx
          y(i) = exp(-(x(i)-5.0)*(x(i)-5.0))
          v(i) = 0.0
      end do
      
      do it=1,nt-1
          dt = t(it+1)-t(it)
          call laplacian(nx,x,y,ylap)
          do i=1,nx
              y(i) = y(i) + dt*v(i)
              v(i) = v(i) + dt*ylap(i)
          end do
      end do


      end program

      subroutine linspace(n,xmin,xmax,x)
      implicit real*8(a-h,o-z),integer*4(i-n)
      real*8 x(n)
      x(1)=xmin
      step = (xmax-xmin)/(n)
      do i=2,n-1
          x(i)=xmin+i*step
      end do
      x(n)=xmax
      end subroutine

      subroutine laplacian(n,x,y,ylap)
      real*8 x(n)
      real*8 y(n)
      real*8 ylap(n)

      ylap(1)=0.0
      ylap(n)=0.0
      dx = x(2)-x(1)
      dx2 = dx*dx
      do i=2,n-1
          ylap(i)=(y(i+1)+y(i-1)-2.0*y(i))/dx2
      end do

      end subroutine
