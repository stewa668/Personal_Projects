%%wave equation in 1D
function wave(endtime)
    %finite 
    nx = 500;
    x = linspace(0.0,10.0,nx);
    nt = 100000*(endtime/10.0);
    t = linspace(0.0,endtime,nt);
    
    %set up initial conditions
    y = exp(-(x-5.0).*(x-5.0));
    v = zeros(size(y));
    
    for it=1:nt-1
        dt=t(it+1)-t(it);
        v = v + dt*laplacian(x,y);
        y = y + dt*v;
    end
    
end
