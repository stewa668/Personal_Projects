    function retval=laplacian(x,y)
        nx = length(x);
        dx=x(2)-x(1);
        retval(1)=0.0;
        for i=2:nx-1;
            retval(i)=(y(i+1)+y(i-1)-2.0*y(i))/(dx^2);
        end
        retval(nx)=0.0;
        return;
    end
