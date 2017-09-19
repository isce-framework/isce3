//
// Author: Piyush Agram
// Copyright 2017
//

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include "isce/core/Projections.h"

namespace isce { namespace core {


/*******************LatLon Projection*****************/

void LatLon::print() const 
{
    std::cout << "Projection: Latlon, EPSG: " << epsgcode << "\n";
}

//This is just a pass through
int LatLon::forward( const std::vector<double>& in,
                        std::vector<double>& out) const
{
    out = in;
    return 0;
}

//This is just a pass through
int LatLon::inverse( const std::vector<double>& in,
                        std::vector<double>& out) const
{
    out = in;
    return 0;
}

/****************End of LatLon Projection************/


/*******************Geocent Projection****************/

void Geocent::print() const 
{
    std::cout << "Projection: Geocent, EPSG: " << epsgcode << "\n";
}

//This is to transform LLH to Geocent
//This is same as LLH_2_XYZ
int Geocent::forward( const std::vector<double>& llh,
                        std::vector<double>& out) const
{

    //Radius of Earth in East direction
    //This is a function of latitude only assuming symmetricity 
    double lat = llh[0];
    double re = ellipse.rEast(lat);

    //Standard parametric representation of circle as function of longitude
    out[0] = (re + llh[2]) * cos(llh[0]) * cos(llh[1]);
    out[1] = (re + llh[2]) * cos(llh[0]) * sin(llh[1]);

    //Parametric representation with Radius adjusted for eccentricity
    out[2] = ((re * (1. - ellipse.e2)) + llh[2]) * sin(llh[0]);
    return 0;
}

int Geocent::inverse( const std::vector<double>& v,
                        std::vector<double>& llh) const
{
    //Lateral distance normalized by major axis
    double p = (pow(v[0], 2) + pow(v[1], 2)) / pow(ellipse.a, 2);

    //Polar distance normalized by minor axis
    double q = ((1. - ellipse.e2) * pow(v[2], 2)) / pow(ellipse.a, 2);


    double r = (p + q - pow(ellipse.e2, 2)) / 6.;
    double s = (pow(ellipse.e2, 2) * p * q) / (4. * pow(r, 3.));
    double t = pow(1. + s + sqrt(s * (2. + s)), (1./3.));
    double u = r * (1. + t + (1. / t));
    double rv = sqrt(pow(u, 2) + (pow(ellipse.e2, 2) * q));
    double w = (ellipse.e2 * (u + rv - q)) / (2. * rv);
    double k = sqrt(u + rv + pow(w, 2)) - w;

    //Radius adjusted for eccentricity
    double d = (k * sqrt(pow(v[0], 2) + pow(v[1], 2))) / (k + ellipse.e2);

    //Latitude is function of z and radius
    llh[0] = atan2(v[2], d);

    //Longitude is function of x and y
    llh[1] = atan2(v[1], v[0]);

    //Height is function of lcation and radius
    llh[2] = ((k + ellipse.e2 - 1.) * sqrt(pow(d, 2) + pow(v[2], 2))) / k;
    return 0;
}

/***************End of Geocent Projection**************/

/****************UTM Projection**********************/

void UTM::print() const
{
    std::cout << "Projection: UTM, Zone: "<< zone ;
    std::cout << ((isnorth)? "N" : "S");
    std::cout << ", EPSG: " << epsgcode << "\n";
};

//Meant to be a private function
//For computation of Gaussian Latitude
double gatg(const double *p1, int len_p1, double B) 
{
    double const *p;
    double h=0.0;
    double h2=0.0;
    double h1, cos_2B;

    cos_2B = 2 * cos(2 * B);
    for(p=p1+len_p1, h1 = *--p; p-p1; h2 = h1, h1 = h)
        h = -h2 + cos_2B*h1 + *--p;

    return (B + h * sin(2*B));
};

//Meant to be a private function
//Real clenshaw summation
double clens(const double *a, int size, double arg_r) 
{
    double const *p;
    double r, hr, hr1, hr2, cos_arg_r;

    p = a + size;
    cos_arg_r  = cos(arg_r);
    r          =  2*cos_arg_r;

    /* summation loop */
    for (hr1 = 0, hr = *--p; a - p;) {
        hr2 = hr1;
        hr1 = hr;
        hr  = -hr2 + r*hr1 + *--p;
    }
    return sin (arg_r)*hr;    
};

//Meant to be a private function
//Complex Clenshaw summation
double clenS(const double *a, int size, double arg_r, double arg_i, double *R, double *I) {
    double const *p;
    double r, i, hr, hr1, hr2, hi, hi1, hi2;
    double      sin_arg_r, cos_arg_r, sinh_arg_i, cosh_arg_i;

    /* arguments */
    p = a + size;
    sin_arg_r  = sin(arg_r);
    cos_arg_r  = cos(arg_r);
    sinh_arg_i = sinh(arg_i);
    cosh_arg_i = cosh(arg_i);
    r          =  2*cos_arg_r*cosh_arg_i;
    i          = -2*sin_arg_r*sinh_arg_i;

    /* summation loop */
    for (hi1 = hr1 = hi = 0, hr = *--p; a - p;) {
        hr2 = hr1;
        hi2 = hi1;
        hr1 = hr;
        hi1 = hi;
        hr  = -hr2 + r*hr1 - i*hi1 + *--p;
        hi  = -hi2 + i*hr1 + r*hi1;
    }

    r   = sin_arg_r*cosh_arg_i;
    i   = cos_arg_r*sinh_arg_i;
    *R  = r*hr - i*hi;
    *I  = r*hi + i*hr;
    return *R;
};

//Meant to be a private function
//Compute log(1+x) accurately
double log1py(double x) {           
    volatile double
      y = 1 + x,
      z = y - 1;
    /* Here's the explanation for this magic: y = 1 + z, exactly, and z
     * approx x, thus log(y)/z (which is nearly constant near z = 0) returns
     * a good approximation to the true log(1 + x)/x.  The multiplication x *
     * (log(y)/z) introduces little additional error. */
    return z == 0 ? x : x * log(y) / z;
}


//Meant to be a private function
//Compute asinh(x) accurately
double asinhy(double x) {              
    double y = abs(x);
    y = log1py(y * (1 + y/(hypot(1.0, y) + 1)));
    return x < 0 ? -y : y;
}


//This is called as part of the Constructor
//This is similar to PROJ.4 setup function for each projection
void UTM::setup() 
{
    //Northern Hemisphere codes are between 32601 and 32660
    if ((epsgcode > 32600) && (epsgcode <= 32660))
    {
        zone = epsgcode - 32600;
        isnorth = true;
    }
    else if ((epsgcode > 32700) && (epsgcode <= 32700))
    {
        zone = epsgcode - 32700;
        isnorth = false;
    }
    else
    {
        throw "Illegal EPSG Code for UTM Projection !!!";
    }

    lon0 = (zone - 0.5) * M_PI/30.0 - M_PI;

    
    //Ellipsoid flattening
    double f = ellipse.e2 / (1 + sqrt(1 - ellipse.e2));

    //Third flattening
    double n = f / (2.0 - f);

    //Gaussian -> Geodetic (cgb)
    //Geodetic -> Gaussian (cbg)
    double np = n;

    cgb[0] = n*( 2 + n*(-2/3.0  + n*(-2      + n*(116/45.0 + n*(26/45.0 +
                n*(-2854/675.0 ))))));
    cbg[0] = n*(-2 + n*( 2/3.0  + n*( 4/3.0  + n*(-82/45.0 + n*(32/45.0 +
                n*( 4642/4725.0))))));

    np *= n;
    cgb[1] = np*(7/3.0 + n*( -8/5.0  + n*(-227/45.0 + n*(2704/315.0 +
                n*( 2323/945.0)))));
    cbg[1] = np*(5/3.0 + n*(-16/15.0 + n*( -13/9.0  + n*( 904/315.0 +
                n*(-1522/945.0)))));

    np *= n;
    cgb[2] = np*( 56/15.0  + n*(-136/35.0 + n*(-1262/105.0 +
                n*( 73814/2835.0))));
    cbg[2] = np*(-26/15.0  + n*(  34/21.0 + n*(    8/5.0   +
                n*(-12686/2835.0))));

    np *= n;
    cgb[3] = np*(4279/630.0 + n*(-332/35.0 + n*(-399572/14175.0)));
    cbg[3] = np*(1237/630.0 + n*( -12/5.0  + n*( -24832/14175.0)));


    np *= n;
    cgb[4] = np*(4174/315.0 + n*(-144838/6237.0 ));
    cbg[4] = np*(-734/315.0 + n*( 109598/31185.0));

    np *= n;
    cgb[5] = np*(601676/22275.0 );
    cbg[5] = np*(444337/155925.0);


    //Computing some ants of the projection
    np = n * n;

    //We have fixed k0 = 0.9996 here. This is standard for WGS84 zones.
    //Proj4 allows this to be changed for custom definitions
    //We plan to support standard definitions only
    Qn = 0.9996/(1 + n) * (1 + np*(1/4.0 + np*(1/64.0 + np/256.0)));
    

    //Elliptical N,E -> Spherical N,E (utg) 
    //Spherical N,E -> Elliptical N,E (gtu)

    utg[0] = n*(-0.5  + n*( 2/3.0 + n*(-37/96.0 + n*( 1/360.0 +
                n*(  81/512.0 + n*(-96199/604800.0))))));
    gtu[0] = n*( 0.5  + n*(-2/3.0 + n*(  5/16.0 + n*(41/180.0 +
                n*(-127/288.0 + n*(  7891/37800.0 ))))));

    utg[1] = np*(-1/48.0 + n*(-1/15.0 + n*(437/1440.0 + n*(-46/105.0 +
                n*( 1118711/3870720.0)))));
    gtu[1] = np*(13/48.0 + n*(-3/5.0  + n*(557/1440.0 + n*(281/630.0 +
                n*(-1983433/1935360.0)))));

    np *= n;
    utg[2] = np*(-17/480.0 + n*(  37/840.0 + n*(  209/4480.0  +
                n*( -5569/90720.0 ))));
    gtu[2] = np*( 61/240.0 + n*(-103/140.0 + n*(15061/26880.0 +
                n*(167603/181440.0))));

    np *= n;
    utg[3] = np*(-4397/161280.0 + n*(  11/504.0 + n*( 830251/7257600.0)));
    gtu[3] = np*(49561/161280.0 + n*(-179/168.0 + n*(6601661/7257600.0)));

    np *= n;
    utg[4] = np*(-4583/161280.0 + n*(  108847/3991680.0));
    gtu[4] = np*(34729/80640.0  + n*(-3418889/1995840.0));

    np *= n;
    utg[5] = np*(-20648693/638668800.0);
    gtu[5] = np*(212378941/319334400.0);

    /*Gaussian latitude of origin latitude*/
    double Z = gatg(cbg, 6, 0.0);

    Zb = -Qn * (Z + clens(gtu, 6, 2*Z));
};


//Transform from LLH to UTM
int UTM::forward( const std::vector<double>& llh,
                    std::vector<double> &utm) const
{
    //Elliptical Lat, Lon -> Gaussian Lat, Lon
    double Cn = gatg(cbg, 6, llh[0]);

    double sin_Cn = sin(Cn);
    double cos_Cn = cos(Cn);

    //Adjust longitude for zone offset
    double sin_Ce = sin(llh[1] - lon0);
    double cos_Ce = cos(llh[1] - lon0);

    //Account for longitude and get Spherical N,E
    Cn = atan2(sin_Cn, cos_Ce * cos_Cn);
    double Ce = atan2(sin_Ce*cos_Cn, hypot(sin_Cn, cos_Cn*cos_Ce));

    //Spherical N,E to Elliptical N,E
    double dCn, dCe;
    Ce = asinhy( tan(Ce) );
    Cn += clenS(gtu,6,2*Cn, 2*Ce, &dCn, &dCe);
    Ce += dCe;

    if (abs(Ce) <= 2.623395162778)
    {
        utm[0] = Qn * Ce + 500000.0;
        double y = Qn * Cn + Zb;
        y += (isnorth)? 0.0:10000000.0;
        utm[1] = y;

        //UTM is lateral projection only. Height is pass through.
        utm[2] = llh[2];
        return 0;
    }
    else
    {
        return 1;
    }
};


//Transform UTM to LLH
int UTM::inverse( const std::vector<double>& utm,
                        std::vector<double>& llh) const
{
    double Cn = utm[1];
    Cn -= (isnorth)?0.0:10000000.0;

    double Ce = utm[0] - 500000.0;
    double dCn, dCe;

    //Normalize N,E to Spherical N,E
    Cn = (Cn - Zb)/Qn;
    Ce = Ce/Qn;


    if (abs(Ce) <= 2.623395162778)
    {
        //N,E to Spherical Lat, Lon
        Cn += clenS(utg,6,2*Cn,2*Ce,&dCn,&dCe);
        Ce += dCe;
        Ce = atan( sinh(Ce));

        double sin_Cn = sin(Cn);
        double cos_Cn = cos(Cn);
        double sin_Ce = sin(Ce);
        double cos_Ce = cos(Ce);

        //Spherical Lat, Lon to Gaussian Lat, Lon
        Ce = atan2(sin_Ce, cos_Ce*cos_Cn);
        Cn = atan2(sin_Cn*cos_Ce, hypot(sin_Ce, cos_Ce*cos_Cn));

        //Gaussian Lat, Lon to Elliptical Lat, Lon
        llh[0] = gatg(cgb, 6, Cn);
        llh[1] = Ce + lon0;
        
        //UTM is a lateral projection only. Height is pass through.
        llh[2] = utm[2];
        return 0;
    }
    else
    {
        return 1;
    }
};

/******************End of UTM Projection************************/
void PolarStereo::print() const 
{
    std::cout << "Projection: ";
    std::cout << ((isnorth)? "North":"South");
    std::cout << " Polar Stereographic, EPSG: " << epsgcode << "\n";
};


//Determine small t from PROJ.4
double pj_tsfn(double phi, double sinphi, double e) 
{
    double denominator;
    sinphi *= e;

    denominator = 1.0 + sinphi;
    return (tan(0.5 * (0.5*M_PI - phi)) / pow((1.0 - sinphi)/(denominator), 0.5*e));
};

//Setup various parameters for polar stereographic projection
//Currently only EPSG:3031 (Antarctic) and EPSG:3413 (Greenland)
//are supported
void PolarStereo::setup()
{
    //Standard Antarctic projection
    if (epsgcode == 3031)
    {
        isnorth = false;
        lat0 = -M_PI/2.0;
        lat_ts = 71.0 * M_PI / 180.0; //Only need abs value
        lon0 = 0.0;
    }
    else if (epsgcode == 3413) //Standard Greenland projection
    {
        isnorth = true;
        lat0 = M_PI/2.0;
        lat_ts = 70.0 * M_PI / 180.0;
        lon0 = -45.0 * M_PI / 180.0;
    }
    else
    {
        throw "Unknown EPSG code for Polar Stereographic projection";
    }

    e = sqrt(ellipse.e2);
    double t = sin(lat_ts);
    akm1 = cos(lat_ts) / pj_tsfn(lat_ts, t, e);
    t *= e;
    akm1 /= sqrt(1.0 - t*t);
};


//Transform from LLH to Polar Stereo
int PolarStereo::forward( const std::vector<double> &llh, 
                                std::vector<double> & out) const
{
    double phi = llh[0] - lat0;
    double lam = llh[1] - lon0;
    double coslam = cos(lam);
    double sinlam = sin(lam);
    double sinphi = sin(phi);

    if (!isnorth)
    {
        phi = -phi;
        coslam = -coslam;
        sinphi = -sinphi;
    }

    double temp = akm1 * pj_tsfn(phi, sinphi, e);

    out[0] = temp * sinlam;
    out[1] = -temp * coslam;

    //Height is just pass through
    out[2] = llh[2];

    return 0;
};


//Transform from Polar Stereo to LLH
int PolarStereo::inverse( const std::vector<double> &ups, 
                                std::vector<double> &llh) const
{
    double rho = hypot(ups[0], ups[1]);
    double tp = -rho / akm1;

    double x = ups[0];
    double y = ups[1];
    if (isnorth) y =- y;

    double phi_l = 0.5*M_PI - 2.0 * atan(tp);
    double halfpi = -0.5*M_PI;
    double halfe = -0.5 * e;

    double phi=0.0;
    double lam=0.0;
    for(int i=8; i--; phi_l = phi)
    {
        double sinphi = e * sin(phi);
        phi = 2.0 * atan(tp * pow((1.0+sinphi)/(1.0-sinphi),halfe) - halfpi);

        if (abs(phi_l - phi) < 1.0e-10)
        {
            if (!isnorth) phi = -phi;
            lam = (x == 0. &&  y == 0.) ? 0. : atan2(x, y);

            llh[0] = phi + lat0;
            llh[1] = lam + lon0;
            llh[2] = ups[2];

            return 0;
        }
    }

    return 1;
};

/****************End of Polar Stereo Projection***************/


/****************Projection Factory*********************/
ProjectionBase* createProj(int epsgcode)
{
    //This is the factory method to return a pointer to transformation object.
    //All our coordinate systems use WGS84 ellipsoid.
    //

    Ellipsoid wgs84(6378137.0, 0.0066943799901);

    //Check for Lat/Lon
    if (epsgcode == 4326)
    {
        return new LatLon{wgs84};
    }
    //Check for Geocentric
    else if (epsgcode == 4978)
    {
        return new Geocent{wgs84};
    }
    //Check if UTM
    else if (epsgcode > 32600 && epsgcode < 32800)
    {
        return new UTM{wgs84,epsgcode};
    }
    //Check if Polar Stereo
    else if (epsgcode == 3031  || epsgcode == 3413)
    {
        return new PolarStereo{wgs84, epsgcode};
    }
    else
    {
        throw "Unknown EPSG code in factory";
    }
}

/*****************End of Projection Factory**************/

/*****************Projection Transformer*****************/
int projTransform(ProjectionBase *in, ProjectionBase* out,
                    const std::vector<double> &inpts,
                    std::vector<double> &outpts)
{
    //Consider case where input and output projections are the same
    if (in->epsgcode == out->epsgcode)
    {
        outpts = inpts;
        return 0;
    }
    //Consider case where input is Lat/Lon
    else if (in->epsgcode == 4326)
    {
        return out->forward(inpts, outpts);
    }
    //Consider case where output is Lat/Lon
    else if (out->epsgcode == 4326)
    {
        return -(out->inverse(inpts, outpts));
    }
    else
    {
        std::vector<double> temp(3);
        int status = in->inverse(inpts, temp);

        if (status != 0)
            return -2;

        status = out->forward(temp, outpts);
        if (status != 0)
            return 2;
    }
    return 0;
};


}}
