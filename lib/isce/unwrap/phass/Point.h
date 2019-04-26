// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
// ------------------------------------------------------------------------------
//  Point class definitions
// ------------------------------------------------------------------------------

#ifndef __POINT_H
#define __POINT_H

#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>

class USPoint {

  friend std::ostream& operator<< (std::ostream&, USPoint);
  public:

    unsigned short x;
    unsigned short y;

  public :

    USPoint () { x = 0; y = 0;}
    USPoint (unsigned short a, unsigned short b) { x = a; y = b; }
    ~USPoint () {}

    USPoint(const USPoint& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
    }
    USPoint& operator=(const USPoint& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
	return *this; 
    }

  // operators :

    USPoint operator() (unsigned short a, unsigned short b);
    /*
    bool   operator == (const USPoint &b) const { 
      if(x == b.x && y == b.y) return true; 
      else return false; 
    } 
    */
    bool operator < (const USPoint &b) const {
      if(x*32768 + y < b.x*32768 + b.y) return true;
      else return false;
    }
    /*
    bool   operator != (const USPoint &b) const { 
      if(x == b.x && y == b.y) return false; 
      else return true; 
    } 
    bool operator <= (const USPoint &b) const {
      if(y <= b.y) return true;
      else return false;
    }
    bool operator > (const USPoint &b) const {
      if(y > b.y) return true;
      else return false;
    }
    bool operator >= (const USPoint &b) const {
      if(y >= b.y) return true;
      else return false;
    }
    */
    
    void set (int a, int b) { x = a; y = b; }
    int get_X() { return x; }
    int get_Y() { return y; }
};


class USF3Point {

  friend std::ostream& operator<< (std::ostream&, USF3Point);

  public:

    unsigned short line;
    unsigned short pixel;
    float    x;

  public :

    USF3Point () { line = 0; pixel = 0; x = 0;}
    USF3Point (unsigned short s_, unsigned short c_, float x_) { line = s_; pixel = c_; x = x_;}
    ~USF3Point () {}
    USF3Point operator() (unsigned short line_, unsigned short pixel_, float x_);

    USF3Point(const USF3Point& source_point) { 
	line = source_point.line; 
	pixel = source_point.pixel; 
	x = source_point.x; 
    }
    USF3Point& operator=(const USF3Point& source_point) { 
	line = source_point.line; 
	pixel = source_point.pixel; 
	x = source_point.x; 
	return *this; 
    }

    bool operator < (const USF3Point &b) const {
      if(line*32768 + pixel < b.line*32768 + b.pixel) return true;
      else return false;
    }
    
    void set(int s_, int c_, float x_) { line = s_; pixel = c_; x = x_;}
};


class USF4Point {

  friend std::ostream& operator<< (std::ostream&, USF4Point);

  public:

    unsigned short s;
    unsigned short c;
    float    x;
    float    y;

  public :

    USF4Point () { s = 0; c = 0; x = 0; y = 0;}
    USF4Point (unsigned short s_, unsigned short c_, float x_, float y_) { s = s_; c = c_; x = x_; y = y_; }
    ~USF4Point () {}
    USF4Point operator() (unsigned short s_, unsigned short c_, float x_, float y_);

    USF4Point(const USF4Point& source_point) { 
	s = source_point.s; 
	c = source_point.c; 
	x = source_point.x; 
	y = source_point.y; 
    }
    USF4Point& operator=(const USF4Point& source_point) { 
	s = source_point.s; 
	c = source_point.c; 
	x = source_point.x; 
	y = source_point.y; 
	return *this; 
    }

    bool operator < (const USF4Point &b) const {
      if(s*32768 + c < b.s*32768 + b.c) return true;
      else return false;
    }
    
    void set(int s_, int c_, float x_, float y_) { s = s_; c = c_; x = x_; y = y_; }
};


class USPointKey {

  friend std::ostream& operator<< (std::ostream&, USPointKey);
  friend std::istream& operator>> (std::istream&, USPointKey &);

  public:

    unsigned short x;
    unsigned short y;
    int            key;

  public :

    USPointKey () { x = 0; y = 0; key = 0; }
    USPointKey (unsigned short a, unsigned short b, int key_) { x = a; y = b; key = key_;}
    ~USPointKey () {}

    USPointKey(const USPointKey& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
	key = source_point.key; 
    }
    USPointKey& operator=(const USPointKey& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
	key = source_point.key; 
	return *this; 
    }
  // operators :

    USPointKey operator() (unsigned short a, unsigned short b, int key_);
    int   operator== (const USPointKey &a);
    bool operator < (const USPointKey &a) const {
      return key > a.key;
    }
    
};


class USPointKeyComp
{
  bool reverse;

public:
  USPointKeyComp(const bool& revparam=false) { reverse = revparam; }
  bool operator() (const USPointKey & first, const USPointKey & second) const
  {
    if (reverse) {
      return (first.key > second.key);
    }
    else return ( first.key < second.key );
  }
};


class Point {

  friend std::ostream& operator<< (std::ostream&, Point);
  friend std::istream& operator>> (std::istream&, Point &);

  public:

    int x;
    int y;

  public :

    Point () { x = 0; y = 0; }
    Point (int a, int b) { x = a; y = b; }
    ~Point () {}

    void set (int a, int b) { x = a; y = b; }
    void get (int& a, int& b) { a = x; b = y; }

    int get_X() { return x; }
    int get_Y() { return y; }

    void set_X(int a)  { x = a; } 
    void set_Y(int b)  { y = b; }

    double magnitude () { return sqrt(x*x + y*y); }

  // operators :

    bool operator < (const Point &b) const {
      return (x + y*1073741824 < b.x + b.y*1073741824);
    }

    Point operator() (int a, int b);
    Point operator+  (const Point &a);
    Point operator-  (const Point &a);
    Point operator*  (int a);
    Point &operator+= (const Point &a);
    Point &operator-= (const Point &a);
    Point operator~  ();
    int   operator*  (const Point &a);
    int   operator== (const Point &a);
    
    friend Point operator* (const int &d, const Point &a);
    
};

class PointKey {

  friend std::ostream& operator<< (std::ostream &stream, PointKey a) { 
    stream << "(" << a.point.x << ", " << a.point.y << ", " << a.key << ")" ;
    return stream;
  }

  public:
    Point point;
    int   key;

  public :

    PointKey () { point = Point(0,0); key = 0; }
    PointKey (int x_, int y_, int key_) { point = Point(x_, y_); key = key_;}
    PointKey (Point point_, int key_) { point = point_; key = key_;}
    ~PointKey () {}

    PointKey(const PointKey& source_point) { 
	point = source_point.point; 
	key = source_point.key; 
    }
    PointKey& operator=(const PointKey& source_point) { 
	point = source_point.point; 
	key = source_point.key; 
	return *this; 
    }
  // operators :

    int   operator == (const PointKey &a) {
      return (point == a.point) && (key = a.key);
    }
    
    bool operator < (const PointKey &a) const {
      return key > a.key;
    }
};

/*
struct PointKeyComp 
{
  bool operator()(const PointKey &k1, const PointKey &k2) const {
    return k1.key < k2.key;
  }
};
*/

/*
class PointComp
{
  bool reverse;

public:
  PointComp(const bool& revparam=false) { reverse = revparam; }
  bool operator() (const Point& first, const Point& second) const
  {
    if (reverse) {
      return (first.y > second.y);
    }
    else return ( first.y < second.y );
  }
};
*/
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

class F2point {

  friend std::ostream& operator<< (std::ostream&, F2point);
  friend std::istream& operator>> (std::istream&, F2point &);

  public:

    float x;
    float y;

  public :

    F2point () { x = 0; y = 0; }
    F2point (float a, float b) { x = a; y = b; }
    F2point (float *a) { x = a[0]; y = a[1]; }
    ~F2point () {}

    F2point(const F2point& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
    }
    F2point& operator=(const F2point& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
	return *this; 
    }
};

class F3point {

  friend std::ostream& operator<< (std::ostream&, F3point);
  friend std::istream& operator>> (std::istream&, F3point &);

  public:

    float x;
    float y;
    float z;

  public :

    F3point () { x = 0; y = 0; z = 0; }
    F3point (float a, float b, float c) { x = a; y = b; z = c; }
    F3point (float *a) { x = a[0]; y = a[1]; z = a[2];}
    ~F3point () {}

    F3point(const F3point& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
	z = source_point.z; 
    }
    F3point& operator=(const F3point& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
	z = source_point.z; 
	return *this; 
    }
};


class D2point {

  friend std::ostream& operator<< (std::ostream&, D2point);
  friend std::istream& operator>> (std::istream&, D2point &);

  public:

    double x;
    double y;

  public :

    D2point () { x = 0; y = 0; }
    D2point (double a, double b) { x = a; y = b; }
    D2point (double *a) { x = a[0]; y = a[1]; }
    ~D2point () {}

    void set (double a, double b) { x = a; y = b; }
    void get (double& a, double& b) { a = x; b = y; }

    double get_X() { return x; }
    double get_Y() { return y; }

    void set_X(double a) { x = a; }
    void set_Y(double a) { y = a; }
	
    double magnitude () { return sqrt(x*x + y*y); }

  // operators :

    D2point operator() (double a, double b);
    D2point operator+  (const D2point &a);
    //    D2point operator+  (D2point const &a);
    D2point operator-  (const D2point &a) const;
    D2point operator* (const double &d);
    D2point operator/ (const double &d);
    D2point &operator+= (const D2point &a);
    D2point &operator-= (const D2point &a);
    D2point operator~  ();
    double  operator*  (const D2point &a);
    int     operator== (const D2point &a) const;

    friend D2point operator*(const double &d, const D2point &a);

    D2point(const D2point& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
    }
    D2point& operator=(const D2point& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
	return *this; 
    }

};


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

//class D3point : public D2point {
class D3point {

  friend std::ostream& operator<< (std::ostream&, D3point);
  friend std::istream& operator>> (std::istream&, D3point &);

  public: // members

    double x;
    double y;
    double z;

  public :

    D3point () { x=0; y=0; z=0; }
    D3point (double a, double b, double c) { x=a; y=b; z=c; }
    D3point (double *a) { x=a[0]; y=a[1]; z=a[2]; }
    ~D3point () {}

    void get (double& a, double& b, double& c) { a=x; b=y; c=z; }
    double get_Z() { return z; }
    void set_Z(double c) { z = c; }
    void get (double& a, double& b) { a = x; b = y; }
    
    double get_x() { return x; }
    double get_y() { return y; }

    D3point unity ();
    double magnitude () const { return sqrt(x*x + y*y + z*z); }
    double magnitude2 () const { return x*x + y*y + z*z; }
    void set(double xpos,double ypos,double zpos) { x=xpos; y=ypos; z=zpos; }
    void set(double xpos,double ypos) { x=xpos; y=ypos; }


    // operators:

    D3point operator() (double a, double b, double c);	// (x,y,z)

    D3point operator+  (const D3point &a);
    D3point operator-  (const D3point &a);
    D3point operator*  (const double  &s);  // vector * scalar
    D3point operator/  (const double  &s);  // divide by a scalar
    D3point operator+= (const D3point &a);
    D3point operator-= (const D3point &a);
    D3point operator*= (const double  &s);        // vector * scalar
    D3point operator/= (const double  &s);        // divide by a scalar
    D3point operator&& (const D3point &a);  // cross product

    double  operator*  (const D3point &a);  // scalar product
    double  operator<  (const D3point &a);  // angle between vectors

    int     operator== (const D3point &a);
    int     operator!= (const D3point &a);

    friend D3point operator*(const double &s, const D3point &a);

    D3point(const D3point& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
	z = source_point.z; 
    }
    D3point& operator=(const D3point& source_point) { 
	x = source_point.x; 
	y = source_point.y; 
	z = source_point.z; 
	return *this; 
    }			      
};


//------------------------------------------------------------------------------

class Geopoint {

  friend std::ostream& operator<< (std::ostream&, Geopoint);
  friend std::istream& operator>> (std::istream&, Geopoint&);

  public:

    double  lat, lon;            // geodetic angles in radians
    double  alt;                 // altitude in meters

  public :

    Geopoint () { lat = 0; lon = 0; alt = 0; } 

    Geopoint ( double Lat, double Lon, double Alt = 0.0 ) { lat = Lat; lon = Lon; alt = Alt; }

    ~Geopoint () {}

    Geopoint operator() ( double   Lat, double   Lon, double  Alt );	// alt in meters
    
    Geopoint& operator= (const Geopoint&);

    void get (double& Lat, double& Lon) { Lat = lat; Lon = lon; }
    void get (double& Lat, double& Lon, double& Alt) { Lat = lat; Lon = lon; Alt = alt; }
    void get (double& Alt) {Alt = alt; }         

    double  Lat () {return lat;}
    double  Lon () {return lon;}
    double  Alt () {return alt;}
    
    void set(double Lat,double Lon,double Alt){
      lat = Lat; lon = Lon; alt = Alt;
    }
    void set_alt(double Alt){alt = Alt;}

    

    Geopoint(const Geopoint& source_point) { 
	lat = source_point.lat; 
	lon = source_point.lon; 
	alt = source_point.alt; 
    }

};

double cross(Point p1, Point p2);
double cross(D2point p1, D2point p2);



#endif
