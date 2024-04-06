public class wave {

    static double [] linspace(double min,double max,int n){
        double [] retval = new double[n];
        double step;
        int i;
    
        retval[0] = min;
        step = (max-min)/(double)(n-1);
        for(i=1;i<n-1;i++) {
            retval[i] = min+(double)i*step;
        }
        retval[n-1] = max;
        return retval;
    }
    
    static void calc_dvdt(int n, double []x, double []y, double []retval) {
        double dx;
        int i;
        dx=x[1]-x[0];
        retval[0]=0.0;
        for(i=1;i<n-1;i++) {
            retval[i]=(y[i+1]+y[i-1]-2.0*y[i])/(dx*dx);
        }
        retval[n-1]=0.0;
        return;
    }
    
    public static void main(String [] args) {
        int nx, nt, nd, i,it;
        double []x;
        double []y;
        double []v;
        double []t;
        double []dvdt;
        double dt;
        
        nx = 500;
        x = linspace(0.0,10.0,nx);
        y = new double[nx];
        nt = 100000;
        nd = 50;
        t = linspace(0.0,10.0,nt);
        dvdt = new double[nx];
        v = new double[nx];
    
        
        for (i=0;i<nx;i++) {
            y[i] = Math.exp(-(x[i]-5.0)*(x[i]-5.0));
            v[i] = 0.0;
        }
        
        for(it=0;it<nt-1;it++) {
            dt=t[it+1]-t[it];
            calc_dvdt(nx,x,y,dvdt);
            for(i=0; i<nx; i++) {
                v[i] = v[i] + dt*dvdt[i];
                y[i] = y[i] + dt*v[i];
            }
        }

 //       for (i=0;i<nx;i++) {
 //           System.out.println(""+x[i]+"    "+y[i]);
 //       }
    
    }
}
