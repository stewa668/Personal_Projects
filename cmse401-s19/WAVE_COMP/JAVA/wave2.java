public class wave2 {

    public static void main(String [] args) {
        // define variables
        int nx, nt;   // number of points, number of timesteps
        int i, it;    // index i (x) and it (t)
        double []x;   // position along wave
        double []y;   // elevation of wave
        double []v;   // speed of wave (y direction)
        double []dvdt;// acceleration of wave (y direction)
        double dx,dt; // spacing in x, t
        double xmin, xmax, tmin, tmax; // range of x, t
        
        // define x array
        nx = 500;
        xmin = 0.0;
        xmax = 10.0;
        dx = (xmax-xmin)/(double)(nx-1); // range divided by # intervals
        x = new double[nx];
        x[0]=xmin;
        for(i=1;i<nx-1;i++) x[i]=xmin+(double)i*dx; // min + i * dx
        x[nx-1]=xmax;

        // define t spacing
        nt = 100000;
        tmin = 0.0;
        tmax = 10.0;
        dt = (tmax-tmin)/(double)(nt-1);
        
        // instantiate y, x, dvdt arrays
        y = new double[nx];
        dvdt = new double[nx];
        v = new double[nx];
        
        // initialize arrays
        //     y is a peak in the middle of the wave
        for (i=0;i<nx;i++) {
            y[i] = Math.exp(-(x[i]-(xmax-xmin)/2.0)*
                     (x[i]-(xmax-xmin)/2.0));
            v[i] = 0.0;
        }
        
        // iterative loop
        for(it=0;it<nt-1;it++) {
            // calculation dvdt at interior positions
            for(i=1;i<nx-1;i++) {
                dvdt[i] = (y[i+1]+y[i-1]-2.0*y[i])/(dx*dx);
            }
            // update v and y
            for(i=0; i<nx; i++) {
                v[i] = v[i] + dt*dvdt[i];
                y[i] = y[i] + dt*v[i];
            }
        }

        // output
        for(i=0;i<nx;i++) {
            System.out.println(""+x[i]+"\t"+y[i]);
        }

    }
}
