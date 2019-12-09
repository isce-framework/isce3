// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 

//------------------------------------------------------------------------------
// Point class methods
//------------------------------------------------------------------------------



#include <math.h>

#include "Point.h"
 

Point Point::operator() (int a, int b)
{
  x = a;
  y = b;
  return *this;
}

//------------------------------------------------------------------------------

Point Point::operator+ (const Point& b) 
{
  Point temp;
  temp.x = x + b.x;
  temp.y = y + b.y;
  return temp;
}

//------------------------------------------------------------------------------

Point Point::operator- (const Point& b) 
{
  Point temp;
  temp.x = x - b.x;
  temp.y = y - b.y;
  return temp;
}

//------------------------------------------------------------------------------


Point Point::operator~ ()
{
  y = -y;
  return *this;
}

//------------------------------------------------------------------------------

Point Point::operator* (int b) 
{
  return Point(x * b, y * b);
}

//------------------------------------------------------------------------------
int Point::operator* (const Point& b) 
{
  return x * b.x + y * b.y;
}

//------------------------------------------------------------------------------
int Point::operator== (const Point& b) 
{
  return (x == b.x) && (y == b.y);
}

//------------------------------------------------------------------------------
Point &Point::operator+= (const Point &a)
{
	x += a.x;
	y += a.y;
	return *this;
}

//------------------------------------------------------------------------------
Point &Point::operator-= (const Point &a)
{
	x -= a.x;
	y -= a.y;
	return *this;
}

//------------------------------------------------------------------------------

std::ostream &operator<< (std::ostream &stream, Point a)
{
  stream << "(" << a.x << ", " << a.y << ")" ;
  return stream;
}

std::istream &operator>> (std::istream &stream, Point &a)
{
  std::cout << "x : ";
  stream  >> a.x ;
  std::cout << "y : ";
  stream  >> a.y ;
  return stream;
}

 
double cross(Point p1, Point p2) {
  D3point d3p1(p1.get_X(), p1.get_Y(), 0.0);
  D3point d3p2(p2.get_X(), p2.get_Y(), 0.0); 
  return (d3p1 && d3p2).magnitude();    
}	
double cross(D2point p1, D2point p2) {
  D3point d3p1(p1.get_X(), p1.get_Y(), 0.0);
  D3point d3p2(p2.get_X(), p2.get_Y(), 0.0); 
  return (d3p1 && d3p2).magnitude();    
}					      	
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------

D2point D2point::operator() (double a, double b)
{
  x = a;
  y = b;
  return *this;
}

//------------------------------------------------------------------------------

D2point D2point::operator+ (const D2point &b) 
{
  D2point temp;
  temp.x = x + b.x;
  temp.y = y + b.y;
  return temp;
}

//------------------------------------------------------------------------------

D2point D2point::operator- (const D2point &b) const
{
  D2point temp;
  temp.x = x - b.x;
  temp.y = y - b.y;
  return temp;
}

//------------------------------------------------------------------------------

D2point D2point::operator* (const double &d) 
{
	D2point temp;
	temp.x = x*d;
	temp.y = y*d;
	return temp;
}
//------------------------------------------------------------------------------

D2point D2point::operator/ (const double &d) 
{
	D2point temp;
	temp.x = x/d;
	temp.y = y/d;
	return temp;
}
//------------------------------------------------------------------------------

D2point &D2point::operator+= (const D2point &a)
{

	x += a.x;
	y += a.y;
	return *this;
}
//------------------------------------------------------------------------------

D2point &D2point::operator-= (const D2point &a)
{
	x -= a.x;
	y -= a.y;
	return *this;
}

//------------------------------------------------------------------------------

D2point D2point::operator~ ()
{
  y = -y;
  return *this;
}

//------------------------------------------------------------------------------

double D2point::operator* (const D2point &b)
{
  return x * b.x + y * b.y;
}

//------------------------------------------------------------------------------

int D2point::operator== (const D2point &b) const
{
  return (x == b.x) && (y == b.y);
}

D2point operator*(const double &d, const D2point &a)
{
	D2point temp;
	temp.x = d*a.x;
	temp.y = d*a.y;
	return temp;
}

std::ostream &operator<< (std::ostream &stream, D2point a)
{
  stream << std::setprecision(12) << "(" << a.x << ", " << a.y << ")";
  return stream;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


D3point D3point::operator() (double a, double b, double c)
{
  x = a;
  y = b;
  z = c;
  return *this;
}


//- vector + vector ------------------------------------------------------------

D3point D3point::operator+ (const D3point &b) 
{
  D3point temp;
  temp.x = x + b.x;
  temp.y = y + b.y;
  temp.z = z + b.z;
  return temp;
}

//- vector - vector ------------------------------------------------------------

D3point D3point::operator- (const D3point &b) 
{
  D3point temp(x - b.x, y - b.y, z - b.z);
  return temp;
}
//- vector * scalar ------------------------------------------------------------

D3point D3point::operator* (const double &s) 
{
  D3point temp;
  temp.x = x * s;
  temp.y = y * s;
  temp.z = z * s;
  return temp;
}

//- vector * scalar ------------------------------------------------------------

D3point operator* (const double &s, const D3point &a)
{
  D3point temp;
  temp.x = a.x * s;
  temp.y = a.y * s;
  temp.z = a.z * s;
  return temp;
}

//- vector / scalar ------------------------------------------------------------

D3point D3point::operator/ (const double &s) 
{
  D3point temp;
  temp.x = x / s;
  temp.y = y / s;
  temp.z = z / s;
  return temp;
}

D3point D3point::operator+= (const D3point &b)
{
  x += b.x;
  y += b.y;
  z += b.z;
  return *this;
}

//- vector - vector ------------------------------------------------------------

D3point D3point::operator-= (const D3point &b)
{
  x -= b.x;
  y -= b.y;
  z -= b.z;
  return *this;
}
//- vector * scalar ------------------------------------------------------------

D3point D3point::operator*= (const double &s)
{
  x *= s;
  y *= s;
  z *= s;
  return *this;
}

//- vector / scalar ------------------------------------------------------------

D3point D3point::operator/= (const double &s)
{
  x /= s;
  y /= s;
  z /= s;
  return *this;
}


//- cross product --------------------------------------------------------------

D3point D3point::operator&& (const D3point &b) 
{
  D3point temp;
  temp.x = y * b.z - z * b.y;
  temp.y = z * b.x - x * b.z;
  temp.z = x * b.y - y * b.x;
  return temp;
}


//- scalar product -------------------------------------------------------------

double D3point::operator* (const D3point &b) 
{
  return x * b.x + y * b.y + z * b.z;
}

D3point D3point::unity ()
{
  double mag = this->magnitude();
  if (mag > 0.0) {
    x /= mag;
    y /= mag;
    z /= mag;
  }
  return *this;
}

//- angle between two vectors --------------------------------------------------

double D3point::operator< (const D3point &b) 
{
  double t = (*this) * b;
  double mag1 = (*this).magnitude();
  double mag2 = sqrt(b.x*b.x + b.y*b.y + b.z*b.z);
  return acos (t/mag1/mag2);
}

//- regular check for identity -------------------------------------------------

int D3point::operator== (const D3point &b) 
{
  return (x == b.x) && (y == b.y) && (z == b.z);
}

int D3point::operator!= (const D3point& b) 
{
  return !((x == b.x) && (y == b.y) && (z == b.z));
}


//------------------------------------------------------------------------------
//- input / output operator overloading ----------------------------------------
//------------------------------------------------------------------------------

std::ostream &operator<< (std::ostream &stream, D3point a)
{
  stream << std::setprecision(12) << "(" << a.x << ", " << a.y << ", " << a.z << ")";
  return stream;
}

std::istream &operator>> (std::istream &stream, D3point &a)
{
  std::cout << "x : ";
  stream  >> a.x ;
  std::cout << "y : ";
  stream  >> a.y ;
  std::cout << "z : ";
  stream  >> a.z ;
  return stream;
}

// Geopoint 
Geopoint Geopoint::operator() ( double   Lat, double   Lon, double  Alt ) {
  lat = Lat;
  lon = Lon;
  alt = Alt;
  return *this;
}

Geopoint& Geopoint::operator= (const Geopoint& p_copy) {    
    if(this != &p_copy) {        
        lat = p_copy.lat;
        lon = p_copy.lon;
        alt = p_copy.alt;
    }
    return *this;
}

std::ostream &operator<< (std::ostream &stream, Geopoint a)
{
  stream << std::setprecision(8)<<"(" <<a.lat<< ", "<<a.lon<< ", " << std::setprecision(12) << a.alt << " m)";
  return stream;
}





USPointKey USPointKey::operator() (unsigned short a, unsigned short b, int key_)
{
  x = a;
  y = b;
  key = key_;
  return *this;
}

//------------------------------------------------------------------------------

USF3Point USF3Point::operator() (unsigned short s_, unsigned short c_, float x_)
{
  line = s_;
  pixel = c_;
  x = x_;
  return *this;
}

std::ostream & operator<< (std::ostream &stream, USF3Point a) {
    stream << "(" << a.line << ", " << a.pixel << ", "<< a.x << ")" ;
    return stream;
}


//------------------------------------------------------------------------------

USF4Point USF4Point::operator() (unsigned short s_, unsigned short c_, float x_, float y_)
{
  s = s_;
  c = c_;
  x = x_;
  y = y_;
  return *this;
}

std::ostream & operator<< (std::ostream &stream, USF4Point a) {
    stream << "(" << a.s << ", " << a.c << ", "<< a.x << ", " << a.y << ")" ;
    return stream;
}

std::ostream &operator<< (std::ostream &stream, USPoint a)
{
  stream << "(" << a.x << ", " << a.y << ")" ;
  return stream;
}

//------------------------------------------------------------------------------
int USPointKey::operator== (const USPointKey& b) 
{
  return (x == b.x) && (y == b.y) && (key = b.key);
}


std::ostream &operator<< (std::ostream &stream, USPointKey a)
{
  stream << "(" << a.x << ", " << a.y << ", " << a.key << ")" ;
  return stream;
}

std::istream &operator>> (std::istream &stream, USPointKey &a)
{
  std::cout << "x : ";
  stream  >> a.x ;
  std::cout << "y : ";
  stream  >> a.y ;
  std::cout << "key : ";
  stream  >> a.key ;

  return stream;
}

 		      	
//------------------------------------------------------------------------------

