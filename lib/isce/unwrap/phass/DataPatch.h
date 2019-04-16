// Copyright (c) 2017-, California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged.
// All rights reserved.
// 
// Author(s):
// 
//  
//  ======================================================================
// 
//  FILENAME: DataPatch.h
// 
// 
//  ======================================================================

#ifndef __DATA_PATCH
#define __DATA_PATCH

#include "stdlib.h"
#include <complex>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
            
//------------------------------------------------------------------------------

using namespace std;

template <class type>
class DataPatch {

  //  friend std::ostream& operator<< (std::ostream&, DataPatch& patch);

  private:

    DataPatch(const DataPatch & patch);
    DataPatch & operator = (const DataPatch & patch);


  protected: 
    int nr_lines;		// maximum line extent of patch
    int nr_pixels;		// maximum pixel extent of patch

    int actual_start_line;	// start of "good" data inside patch
    int actual_nr_lines;	// number of "good" data lines inside patch

    int actual_start_pixel;	// start of "good" pixels inside patch
    int actual_nr_pixels;	// number of "good" data pixels inside patch

    int extern_start_line;	// corresponding to actual_start_line in the file
    int extern_start_pixel;	// corresponding to actual_start_pixel in the file

    type  *data;		// pointer to the data
    type **data_lines;		// pointer to array of line pointers

  public:

    //DataPatch();     //added by craig
    DataPatch(int pixels, int lines);
    
    virtual type operator() (int x,int y);
    
    virtual type operator[] (int k);
    virtual const type operator[] (int k) const;
    
    virtual ~DataPatch();


    void set_actual_lines (int act_start_line, int act_nr_lines);

    void set_actual_pixels (int act_start_pixel, int act_nr_pixels);

    void set_extern_start_line (int start_line);

    void set_extern_start_pixel (int start_pixel);

    int get_nr_lines ()           {return nr_lines;}
    int get_nr_pixels ()          {return nr_pixels;}
    int get_actual_start_line ()  {return actual_start_line;}
    int get_actual_nr_lines ()    {return actual_nr_lines;}
    int get_actual_start_pixel () {return actual_start_pixel;}
    int get_actual_nr_pixels ()   {return actual_nr_pixels;}
    int get_extern_start_line ()  {return extern_start_line;}
    int get_extern_start_pixel () {return extern_start_pixel;}

    type *get_data_ptr ()         {return data;}
    type **get_data_lines_ptr ()  {return data_lines;}

    type mean();

    void dump (char *filename);    // dumps the used part of patch to filename
    void dumpall(char *filename);
    void write (char *filename, int append=0);    // dumps the used part of patch to filename
    
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

template <class type> inline type DataPatch<type>::operator() (int y,int x) {
  //  DPRINT("operator()\n");
  if(y<0 || y>=nr_lines)
    cerr << "DataPatch::operator()(int x,int y), Invalid par y: " << y << endl;
  if(x<0 || x>=nr_pixels)
    cerr << "DataPatch::operator()(int x,int y), Invalid par x: " << x << endl;
  return data_lines[y][x];
}
  
template <class type> inline type DataPatch<type>::operator[] (int k) {  
  //  DPRINT("operator[]\n");
  if(k < 0 || k >=nr_lines*nr_pixels)
    cerr << "DataPatch::operator[](int k), Invalid par k: " << k << endl;  
  return data[k];
}

template <class type> inline const type DataPatch<type>::operator[] (int k) const {
  //  DPRINT("operator[]\n");
  if(k < 0 || k >=nr_lines*nr_pixels)
    cerr << "DataPatch::operator[](int k), Invalid par k: " << k << endl;  
  return data[k];
}


/*
template <class type> 
inline DataPatch<type>::DataPatch()
{
  //  DPRINT("DataPatch::DataPatch()\n");
  nr_lines           = 0;
  nr_pixels          = 0;

  actual_start_line  = 0;
  actual_nr_lines    = 0;

  actual_start_pixel = 0;
  actual_nr_pixels   = 0;

  extern_start_line  = 0;
  extern_start_pixel = 0;

  data        = NULL;	
  data_lines  = NULL;

  }*/
template <class type> 
inline DataPatch<type>::DataPatch (int pixels, int lines)
{
  //  DPRINT("DataPatch::DataPatch(%d,%d)\n",pixels,lines);
  nr_lines           = lines;
  nr_pixels          = pixels;

  actual_start_line  = 0;
  actual_nr_lines    = nr_lines;

  actual_start_pixel = 0;
  actual_nr_pixels   = nr_pixels;

  extern_start_line  = 0;
  extern_start_pixel = 0;

  data        = NULL;	
  data_lines  = NULL;

  if (nr_lines <= 0)
    cerr << "DataPatch:invalid nr_lines: " << nr_lines << endl;
  if (nr_pixels <= 0)
    cerr << "DataPatch:invalid nr_pixels: " << nr_lines << endl;

  data = new type[nr_lines*nr_pixels];
  data_lines = new type *[nr_lines];

  if (data == NULL)
    cerr <<"DataPatch: unable to allocate memory: " << sizeof(type)*nr_lines*nr_pixels << endl;

  if (data_lines == NULL)
    cerr << "DataPatch: unable to allocate memory: " << sizeof(type)*nr_lines << endl;

  for(int ii = 0; ii < nr_lines; ii++)
    data_lines[ii] = data+ii*nr_pixels;
}

//------------------------------------------------------------------------------

template <class type>
inline DataPatch<type>::~DataPatch()
{
  //  DPRINT("DataPatch::~DataPatch()")
  if(data != NULL) delete [] data;
  if(data_lines != NULL) delete [] data_lines; 
}

//------------------------------------------------------------------------------

template <class type> inline
void DataPatch<type>::set_actual_lines (int act_start_line, int act_nr_lines)
{
  if (act_start_line < 0)
    cerr << "DataPatch:invalid actual_start_line: " << act_start_line << endl;
  if (act_nr_lines <= 0)
    cerr << "DataPatch:invalid actual_nr_lines: " << act_nr_lines << endl;
  if (act_start_line + act_nr_lines > nr_lines)
    cerr << "DataPatch:invalid actual_lines: " << act_start_line+act_nr_lines << endl;
  actual_start_line = act_start_line;
  actual_nr_lines   = act_nr_lines;
}

//------------------------------------------------------------------------------

template <class type> inline
void DataPatch<type>::set_actual_pixels (int act_start_pixel, int act_nr_pixels)
{
  if (act_start_pixel < 0)
    cerr << "DataPatch:invalid actual_start_pixel: " << act_start_pixel << endl;
  if (act_nr_pixels <= 0)
    cerr << "DataPatch:invalid actual_nr_pixels: " << act_nr_pixels << endl;
  if (act_start_pixel + act_nr_pixels > nr_pixels)
    cerr << "DataPatch:invalid actual_pixels: " << act_start_pixel+act_nr_pixels << endl;
  actual_start_pixel = act_start_pixel;
  actual_nr_pixels   = act_nr_pixels;
}

//------------------------------------------------------------------------------

template <class type> inline
void DataPatch<type>::set_extern_start_line (int start_line) 
{
  if (start_line < 0)
    cerr << "DataPatch:invalid extern_start_line: " << start_line << endl;
  extern_start_line  = start_line;
}

//------------------------------------------------------------------------------

template <class type> inline
void DataPatch<type>::set_extern_start_pixel (int start_pixel) 
{
  if (start_pixel < 0)
    cerr << "DataPatch:invalid extern_start_pixel: "<< start_pixel << endl;
  extern_start_pixel = start_pixel;
}

//------------------------------------------------------------------------------
/*
template <class type>
inline std::ostream &operator<< (std::ostream &stream, DataPatch<type>& patch)
{
  stream << "nr_lines                  " << patch.nr_lines  << std::endl;
  stream << "nr_pixels                 " << patch.nr_pixels << std::endl;
  stream << "actual_start_line         " << patch.actual_start_line  << std::endl;
  stream << "actual_start_pixel        " << patch.actual_start_pixel << std::endl;
  stream << "actual_nr_lines           " << patch.actual_nr_lines  << std::endl;
  stream << "actual_nr_pixels          " << patch.actual_nr_pixels << std::endl;
  return stream;
}
*/
//------------------------------------------------------------------------------
template <class type> inline void DataPatch<type>::dump (char *filename)

{
  FILE *fp;
  if (!(fp = fopen(filename, "w"))) {
    cerr << "DataPatch::dump: error on opening: " << filename << endl;
    return;
  }

  // dump the content to file :
  for (int line = actual_start_line; line < actual_nr_lines; line++) {
    for (int pixel = actual_start_pixel; pixel < actual_nr_pixels; pixel++)
      if (fwrite ((void*)&data_lines [line][pixel], sizeof(type), 1, fp) != 1) {
        cerr << "DataPatch::dump: error on writing: " << filename << endl;
        return;
      }
  }
  cerr << "DataPatch::dump: number of pixels: " << actual_nr_pixels << endl;
}
//------------------------------------------------------------------------------
template <class type> inline void DataPatch<type>::write (char *filename, int append)

{
  FILE *fp;
  if(append){
    if (!(fp = fopen(filename, "a"))) {
      cerr << "DataPatch::write: error on opening: " << filename << endl;
      return;
    }
  }else{
    if (!(fp = fopen(filename, "w"))) {
      cerr << "DataPatch::write: error on opening: " << filename << endl;
      return;
    }
  }

  // write the content to file :
  for (int line = actual_start_line; line < actual_nr_lines; line++) {
    for (int pixel = actual_start_pixel; pixel < actual_nr_pixels; pixel++)
      if (fwrite ((void*)&data_lines [line][pixel], sizeof(type), 1, fp) != 1) {
        cerr << "DataPatch::write: error on writing: " << filename << endl;
        return;
      }
  }
  //cerr << "DataPatch::write: number of pixels: " << actual_nr_pixels << endl;
  fclose(fp);
}


//------------------------------------------------------------------------------
template <class type> inline void DataPatch<type>::dumpall (char *filename)
{
  FILE *fp;
  if (!(fp = fopen(filename, "w"))) {
    cerr << "DataPatch::dump: error on opening: " << filename << endl;
    return;
  }

  // dump the content to file :
  for (int line = 0; line < nr_lines; line++) {
    for (int pixel = 0; pixel < nr_pixels; pixel++)
      if (fwrite ((void*)&data_lines [line][pixel], sizeof(type), 1, fp) != 1) {
        cerr << "DataPatch::dump: error on writing: " << filename << endl;
        return;
      }
  }
  cerr << "DataPatch::dumpall: number of pixels: " <<  nr_pixels << endl;
}

//------------------------------------------------------------------------------
// STATISTIC FUNCTIONS
//------------------------------------------------------------------------------

template<class type> inline type DataPatch<type>::mean(){
  int cnt = 0;
  type sum = 0.0;
  for (int line = actual_start_line; line < actual_start_line + actual_nr_lines; line++) {
    for (int pixel = actual_start_pixel; pixel < actual_start_pixel+actual_nr_pixels; pixel++) {
      sum += data_lines [line][pixel];
      cnt++;
    }
  }
  if (cnt != 0) sum /= cnt;
  return sum;
}

//------------------------------------------------------------------------------
#endif
//------------------------------------------------------------------------------
