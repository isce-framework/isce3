/*************************************************************************

  snaphu input/output source file
  Written by Curtis W. Chen
  Copyright 2002 Board of Trustees, Leland Stanford Jr. University
  Please see the supporting documentation for terms of use.
  No warranty.

*************************************************************************/

#include <cstring>
#include <cctype>
#include <unistd.h>
#include <sys/stat.h>

#include <isce3/except/Error.h>

#include "snaphu.h"

namespace isce3::unwrap {

/* static (local) function prototypes */
static
int ParseConfigLine(char *buf, const char *conffile, long nlines,
                    infileT *infiles, outfileT *outfiles,
                    long *linelenptr, paramT *params);
static
int LogStringParam(FILE *fp, const char *key, char *value);
static
int LogBoolParam(FILE *fp, const char *key, signed char boolvalue);
static
int LogFileFormat(FILE *fp, const char *key, signed char fileformat);
static
int WriteAltLineFile(Array2D<float>& mag,
                     Array2D<float>& phase,
                     char *outfile, long nrow, long ncol);
static
int WriteAltSampFile(Array2D<float>& arr1,
                     Array2D<float>& arr2,
                     char *outfile, long nrow, long ncol);



/* function: SetDefaults()
 * -----------------------
 * Sets all parameters to their initial default values.
 */
int SetDefaults(infileT *infiles, outfileT *outfiles, paramT *params){

  /* initialize to start for extra robustness */
  *infiles={};
  *outfiles={};
  *params={};
  
  /* input files */
  StrNCopy(infiles->weightfile,DEF_WEIGHTFILE,MAXSTRLEN);
  StrNCopy(infiles->corrfile,DEF_CORRFILE,MAXSTRLEN);
  StrNCopy(infiles->ampfile,DEF_AMPFILE,MAXSTRLEN);
  StrNCopy(infiles->ampfile2,DEF_AMPFILE2,MAXSTRLEN);
  StrNCopy(infiles->estfile,DEF_ESTFILE,MAXSTRLEN);  
  StrNCopy(infiles->magfile,DEF_MAGFILE,MAXSTRLEN);
  StrNCopy(infiles->costinfile,DEF_COSTINFILE,MAXSTRLEN);
  StrNCopy(infiles->bytemaskfile,DEF_BYTEMASKFILE,MAXSTRLEN);
  StrNCopy(infiles->dotilemaskfile,DEF_DOTILEMASKFILE,MAXSTRLEN);

  /* output and dump files */
  StrNCopy(outfiles->initfile,DEF_INITFILE,MAXSTRLEN);
  StrNCopy(outfiles->flowfile,DEF_FLOWFILE,MAXSTRLEN);
  StrNCopy(outfiles->eifile,DEF_EIFILE,MAXSTRLEN);
  StrNCopy(outfiles->rowcostfile,DEF_ROWCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->colcostfile,DEF_COLCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->mstrowcostfile,DEF_MSTROWCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->mstcolcostfile,DEF_MSTCOLCOSTFILE,MAXSTRLEN);
  StrNCopy(outfiles->mstcostsfile,DEF_MSTCOSTSFILE,MAXSTRLEN);
  StrNCopy(outfiles->corrdumpfile,DEF_CORRDUMPFILE,MAXSTRLEN);
  StrNCopy(outfiles->rawcorrdumpfile,DEF_RAWCORRDUMPFILE,MAXSTRLEN);
  StrNCopy(outfiles->costoutfile,DEF_COSTOUTFILE,MAXSTRLEN);
  StrNCopy(outfiles->conncompfile,DEF_CONNCOMPFILE,MAXSTRLEN);
  StrNCopy(outfiles->outfile,DEF_OUTFILE,MAXSTRLEN);  
  StrNCopy(outfiles->logfile,DEF_LOGFILE,MAXSTRLEN);

  /* file formats */
  infiles->infileformat=DEF_INFILEFORMAT;
  infiles->unwrappedinfileformat=DEF_UNWRAPPEDINFILEFORMAT;
  infiles->magfileformat=DEF_MAGFILEFORMAT;
  infiles->corrfileformat=DEF_CORRFILEFORMAT;
  infiles->estfileformat=DEF_ESTFILEFORMAT;
  infiles->ampfileformat=DEF_AMPFILEFORMAT;
  outfiles->outfileformat=DEF_OUTFILEFORMAT;

  /* options and such */
  params->unwrapped=DEF_UNWRAPPED;
  params->regrowconncomps=DEF_REGROWCONNCOMPS;
  params->eval=DEF_EVAL;
  params->initonly=DEF_INITONLY;
  params->initmethod=DEF_INITMETHOD;
  params->costmode=DEF_COSTMODE;
  params->amplitude=DEF_AMPLITUDE;
  params->verbose=DEF_VERBOSE;

  /* SAR and geometry parameters */
  params->orbitradius=DEF_ORBITRADIUS;
  params->altitude=DEF_ALTITUDE;
  params->earthradius=DEF_EARTHRADIUS;
  params->bperp=DEF_BPERP; 
  params->transmitmode=DEF_TRANSMITMODE;
  params->baseline=DEF_BASELINE;
  params->baselineangle=DEF_BASELINEANGLE;
  params->nlooksrange=DEF_NLOOKSRANGE;
  params->nlooksaz=DEF_NLOOKSAZ;
  params->nlooksother=DEF_NLOOKSOTHER;
  params->ncorrlooks=DEF_NCORRLOOKS;           
  params->ncorrlooksrange=DEF_NCORRLOOKSRANGE;
  params->ncorrlooksaz=DEF_NCORRLOOKSAZ;
  params->nearrange=DEF_NEARRANGE;         
  params->dr=DEF_DR;               
  params->da=DEF_DA;               
  params->rangeres=DEF_RANGERES;         
  params->azres=DEF_AZRES;            
  params->lambda=DEF_LAMBDA;           

  /* scattering model parameters */
  params->kds=DEF_KDS;
  params->specularexp=DEF_SPECULAREXP;
  params->dzrcritfactor=DEF_DZRCRITFACTOR;
  params->shadow=DEF_SHADOW;
  params->dzeimin=DEF_DZEIMIN;
  params->laywidth=DEF_LAYWIDTH;
  params->layminei=DEF_LAYMINEI;
  params->sloperatiofactor=DEF_SLOPERATIOFACTOR;
  params->sigsqei=DEF_SIGSQEI;

  /* decorrelation model parameters */
  params->drho=DEF_DRHO;
  params->rhosconst1=DEF_RHOSCONST1;
  params->rhosconst2=DEF_RHOSCONST2;
  params->cstd1=DEF_CSTD1;
  params->cstd2=DEF_CSTD2;
  params->cstd3=DEF_CSTD3;
  params->defaultcorr=DEF_DEFAULTCORR;
  params->rhominfactor=DEF_RHOMINFACTOR;

  /* pdf model parameters */
  params->dzlaypeak=DEF_DZLAYPEAK;
  params->azdzfactor=DEF_AZDZFACTOR;
  params->dzeifactor=DEF_DZEIFACTOR;
  params->dzeiweight=DEF_DZEIWEIGHT;
  params->dzlayfactor=DEF_DZLAYFACTOR;
  params->layconst=DEF_LAYCONST;
  params->layfalloffconst=DEF_LAYFALLOFFCONST;
  params->sigsqshortmin=DEF_SIGSQSHORTMIN;
  params->sigsqlayfactor=DEF_SIGSQLAYFACTOR;
  
  /* deformation mode parameters */
  params->defoazdzfactor=DEF_DEFOAZDZFACTOR;
  params->defothreshfactor=DEF_DEFOTHRESHFACTOR;
  params->defomax=DEF_DEFOMAX;
  params->sigsqcorr=DEF_SIGSQCORR;
  params->defolayconst=DEF_DEFOLAYCONST;

  /* algorithm parameters */
  params->flipphasesign=DEF_FLIPPHASESIGN;
  params->onetilereopt=DEF_ONETILEREOPT;
  params->rmtileinit=DEF_RMTILEINIT;
  params->initmaxflow=DEF_INITMAXFLOW;
  params->arcmaxflowconst=DEF_ARCMAXFLOWCONST;
  params->maxflow=DEF_MAXFLOW;
  params->krowei=DEF_KROWEI;
  params->kcolei=DEF_KCOLEI;   
  params->kperpdpsi=DEF_KPERPDPSI;
  params->kpardpsi=DEF_KPARDPSI;
  params->threshold=DEF_THRESHOLD;  
  params->initdzr=DEF_INITDZR;    
  params->initdzstep=DEF_INITDZSTEP;    
  params->maxcost=DEF_MAXCOST;
  params->costscale=DEF_COSTSCALE;      
  params->costscaleambight=DEF_COSTSCALEAMBIGHT;      
  params->dnomincangle=DEF_DNOMINCANGLE;
  params->srcrow=DEF_SRCROW;
  params->srccol=DEF_SRCCOL;
  params->p=DEF_P;
  params->bidirlpn=DEF_BIDIRLPN;
  params->nshortcycle=DEF_NSHORTCYCLE;
  params->maxnewnodeconst=DEF_MAXNEWNODECONST;
  params->maxcyclefraction=DEF_MAXCYCLEFRACTION;
  params->nconnnodemin=DEF_NCONNNODEMIN;
  params->maxnflowcycles=DEF_MAXNFLOWCYCLES;
  params->dumpall=DEF_DUMPALL;
  params->nmajorprune=DEF_NMAJORPRUNE;
  params->prunecostthresh=DEF_PRUNECOSTTHRESH;
  params->edgemasktop=DEF_EDGEMASKTOP;
  params->edgemaskbot=DEF_EDGEMASKBOT;
  params->edgemaskleft=DEF_EDGEMASKLEFT;
  params->edgemaskright=DEF_EDGEMASKRIGHT;
  params->parentpid=(long )getpid();


  /* tile parameters */
  params->ntilerow=DEF_NTILEROW;
  params->ntilecol=DEF_NTILECOL;
  params->rowovrlp=DEF_ROWOVRLP;
  params->colovrlp=DEF_COLOVRLP;
  params->piecefirstrow=DEF_PIECEFIRSTROW;
  params->piecefirstcol=DEF_PIECEFIRSTCOL;
  params->piecenrow=DEF_PIECENROW;
  params->piecencol=DEF_PIECENCOL;
  params->tilecostthresh=DEF_TILECOSTTHRESH;
  params->minregionsize=DEF_MINREGIONSIZE;
  params->nthreads=DEF_NTHREADS;
  params->scndryarcflowmax=DEF_SCNDRYARCFLOWMAX;
  StrNCopy(params->tiledir,DEF_TILEDIR,MAXSTRLEN);
  params->assembleonly=DEF_ASSEMBLEONLY;
  params->rmtmptile=DEF_RMTMPTILE;
  params->tileedgeweight=DEF_TILEEDGEWEIGHT;

  /* connected component parameters */
  params->minconncompfrac=DEF_MINCONNCOMPFRAC;
  params->conncompthresh=DEF_CONNCOMPTHRESH;
  params->maxncomps=DEF_MAXNCOMPS;
  params->conncompouttype=DEF_CONNCOMPOUTTYPE;

  /* done */
  return(0);

}


/* function: CheckParams()
 * -----------------------
 * Checks all parameters to make sure they are valid.  This is just a boring
 * function with lots of checks in it.
 */
int CheckParams(infileT *infiles, outfileT *outfiles, 
                long linelen, long nlines, paramT *params){

  long ni, nj, n;
  FILE *fp;

  /* make sure output file is writable (try opening in append mode) */
  /* file will be opened in write mode later, clobbering existing file */
  if((fp=fopen(outfiles->outfile,"a"))==NULL){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "File " + std::string(outfiles->outfile) + " is not writable");
  }else{
    if(ftell(fp)){
      fclose(fp);
    }else{
      fclose(fp);
      remove(outfiles->outfile);
    }
    if(!strcmp(outfiles->outfile,infiles->infile) 
       && !params->eval && !params->regrowconncomps){
      fflush(NULL);
      fprintf(sp0,"WARNING: output will overwrite input\n");
    }
  }

  /* make sure options aren't contradictory */
  if(params->initonly && params->unwrapped){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Cannot use initialize-only mode with unwrapped input");
  }
  if(params->initonly && params->p>=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Cannot use initialize-only mode with Lp costs");
  }
  if(params->costmode==NOSTATCOSTS && !(params->initonly || params->p>=0)){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "no-statistical-costs option can only be used in initialize-only "
            "or Lp-norm modes");
  }
  if(strlen(infiles->costinfile) && params->costmode==NOSTATCOSTS){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "no-statistical-costs option cannot be given if input cost file "
            "is specified");
  }
  if(strlen(outfiles->costoutfile) && params->costmode==NOSTATCOSTS){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "no-statistical-costs option cannot be given if output cost file "
            "is specified");
  }

  /* check geometry parameters */
  if(params->earthradius<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Earth radius must be nonnegative");
  }
  if(params->altitude){
    if(params->altitude>0){
      params->orbitradius=params->earthradius+params->altitude;
    }else{
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Platform altitude must be positive");
    }
  }else if(params->orbitradius < params->earthradius){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Platform orbit radius must be greater than earth radius");
  }
  if(params->costmode==TOPO && params->baseline<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Baseline length must be nonnegative\n");
  }
  if(params->costmode==TOPO && params->baseline==0){
    fflush(NULL);
    fprintf(sp0,"WARNING: zero baseline may give unpredictable results\n");
  }
  if(params->ncorrlooks<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Number of looks ncorrlooks must be positive\n");
  }
  if(params->nearrange<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Slant range parameter nearrange must be positive (meters)");
  }
  if(params->dr<=0 || params->da<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Pixel spacings dr and da must be positive (meters)");
  }
  /* dr and da after multilooking can be larger than rangeres, azres */
  /*
  if(params->rangeres<=(params->dr) 
     || params->azres<=(params->da)){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Resolutions parameters must be larger than pixel spacings");
  }
  */
  if(params->lambda<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Wavelength lambda  must be positive (meters)");
  }

  /* check scattering model defaults */
  if(params->kds<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Scattering model parameter kds must be positive");
  }
  if(params->specularexp<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Scattering model parameter SPECULAREXP must be positive");
  }
  if(params->dzrcritfactor<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "dzrcritfactor must be nonnegative");
  }
  if(params->laywidth<1){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Layover window width laywidth must be positive");
  }
  if(params->layminei<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Layover minimum brightness must be nonnegative");
  }
  if(params->sloperatiofactor<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Slope ratio fudge factor must be nonnegative");
  }
  if(params->sigsqei<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Intensity estimate variance must be positive");
  }

  /* check decorrelation model defaults */
  if(params->drho<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Correlation step size drho must be positive");
  }
  if(params->rhosconst1<=0 || params->rhosconst2<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameters rhosconst1 and rhosconst2 must be positive");
  }
  if(!strlen(infiles->corrfile) 
     && (params->defaultcorr<0 || params->defaultcorr>1)){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Default correlation must be between 0 and 1");
  }
  if(params->rhominfactor<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter rhominfactor must be nonnegative");
  }
  if(params->ncorrlooksaz<1 || params->ncorrlooksrange<1
     || params->nlooksaz<1 || params->nlooksrange<1
     || params->nlooksother<1){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Numbers of looks must be positive integer");
  }
  if(!strlen(infiles->corrfile)){
    if(params->ncorrlooksaz<params->nlooksaz){ 
      fflush(NULL);
      fprintf(sp0,"NCORRLOOKSAZ cannot be smaller than NLOOKSAZ\n");
      fprintf(sp0,"  setting NCORRLOOKSAZ to equal NLOOKSAZ\n");
      params->ncorrlooksaz=params->nlooksaz;
    }
    if(params->ncorrlooksrange<params->nlooksrange){ 
      fflush(NULL);
      fprintf(sp0,"NCORRLOOKSRANGE cannot be smaller than NLOOKSRANGE\n");
      fprintf(sp0,"  setting NCORRLOOKSRANGE to equal NLOOKSRANGE\n");
      params->ncorrlooksrange=params->nlooksrange;
    }
  }
    
  /* check pdf model parameters */
  if(params->azdzfactor<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter azdzfactor must be nonnegative");
  }
  if(params->dzeifactor<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter dzeifactor must be nonnegative");
  }
  if(params->dzeiweight<0 || params->dzeiweight>1.0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter dzeiweight must be between 0 and 1");
  }
  if(params->dzlayfactor<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter dzlayfactor must be nonnegative");
  }
  if(params->layconst<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter layconst must be positive");
  }
  if(params->layfalloffconst<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter layfalloffconst must be nonnegative");
  }
  if(params->sigsqshortmin<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter sigsqshortmin must be positive");
  }
  if(params->sigsqlayfactor<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter sigsqlayfactor must be nonnegative");
  }

  /* check deformation mode parameters */
  if(params->defoazdzfactor<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter defoazdzfactor must be nonnegative");
  }
  if(params->defothreshfactor<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter defothreshfactor must be nonnegative");
  }
  if(params->defomax<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter defomax must be nonnegative");
  }
  if(params->sigsqcorr<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter sigsqcorr must be nonnegative");
  }
  if(params->defolayconst<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Parameter defolayconst must be positive");
  }
  
  /* check algorithm parameters */
  /* be sure to check for things that will cause type overflow */
  /* or floating point exception */
  if((params->initmaxflow)<1 && (params->initmaxflow)!=AUTOCALCSTATMAX){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Initialization maximum flow must be positive");
  }
  if((params->arcmaxflowconst)<1){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "arcmaxflowconst must be positive");
  }
  if((params->maxflow)<1){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "maxflow must be positive");
  }
  if(params->krowei<=0 || params->kcolei<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Averaging window sizes krowei and kcolei must be positive");
  }
  if(params->kperpdpsi<=0 || params->kpardpsi<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
          "Averaging window sizes kperpdpsi and kpardpsi must be positive");
  }
  if(params->threshold<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Numerical solver threshold must be positive");
  }
  if(params->initdzr<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "initdzr must be positive");
  }
  if(params->initdzstep<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "initdzstep must be positive");
  }
  if(params->maxcost>POSSHORTRANGE || params->maxcost<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "maxcost must be positive and within range or short int");
  }
  if(params->costscale<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "cost scale factor costscale must be positive");
  }
  if(params->p<0 && params->p!=PROBCOSTP){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Lp-norm parameter p should be nonnegative");
  }
  if(params->costmode==TOPO && (params->maxflow*params->nshortcycle)
     >POSSHORTRANGE){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "maxflow exceeds range of short int for given nshortcycle");
  }
  if(params->costmode==DEFO && ceil(params->defomax*params->nshortcycle)
     >POSSHORTRANGE){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "defomax exceeds range of short int for given nshortcycle");
  }
  if(params->nshortcycle < 1 || params->nshortcycle > MAXNSHORTCYCLE){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Illegal value for nshortcycle");
  }
  if(params->maxnewnodeconst<=0 || params->maxnewnodeconst>1){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "maxnewnodeconst must be between 0 and 1");
  }
  if(params->nconnnodemin<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "nconnnodemin must be nonnegative");
  }
  if(infiles->infileformat!=FLOAT_DATA || strlen(infiles->magfile)){
    params->havemagnitude=TRUE;
  }else{
    params->havemagnitude=FALSE;
  }
  if(params->maxnflowcycles==USEMAXCYCLEFRACTION){
    params->maxnflowcycles=LRound(params->maxcyclefraction
                                   *nlines/(double )params->ntilerow
                                   *linelen/(double )params->ntilecol);
  }
  if(params->initmaxflow==AUTOCALCSTATMAX 
     && !(params->ntilerow==1 && params->ntilecol==1)){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Initial maximum flow cannot be calculated automatically in "
            "tile mode");
  }

  /* masking parameters */
  if(strlen(infiles->bytemaskfile) 
     || params->edgemasktop || params->edgemaskbot  
     || params->edgemaskleft || params->edgemaskright){
    if(params->initonly){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Masking not applicable for initialize-only mode");
    }
  }
  if(params->edgemasktop<0 || params->edgemaskbot<0  
     || params->edgemaskleft<0 || params->edgemaskright<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "edgemask parameters cannot be negative");
  }
  if(params->edgemasktop+params->edgemaskbot>=nlines  
     || params->edgemaskleft+params->edgemaskright>=linelen){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Edge masks cannot exceed input array size");
  }

  /* tile parameters */
  if(params->ntilerow<1 || params->ntilecol<1){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Numbers of tile rows and columns must be positive");
  }
  if(params->rowovrlp<0 || params->colovrlp<0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Tile overlaps must be nonnegative");
  }
  if(params->ntilerow>1 || params->ntilecol>1){
    ni=ceil((nlines+(params->ntilerow-1)*params->rowovrlp)
            /(double )params->ntilerow);
    nj=ceil((linelen+(params->ntilecol-1)*params->colovrlp)
            /(double )params->ntilecol);
    if(params->ntilerow+params->rowovrlp > nlines 
       || params->ntilecol+params->colovrlp > linelen
       || params->ntilerow*params->ntilerow > nlines
       || params->ntilecol*params->ntilecol > linelen){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Tiles too small or overlap too large for given input");
    }
    if(params->minregionsize 
       > ((nlines-(params->ntilerow-1)*(ni-params->rowovrlp))
          *(linelen-(params->ntilecol-1)*(nj-params->colovrlp)))){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Minimum region size too large for given tile parameters");
    }
    if(TMPTILEOUTFORMAT!=ALT_LINE_DATA && TMPTILEOUTFORMAT!=FLOAT_DATA){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Unsupported TMPTILEOUTFORMAT value in complied binary");
    }
    if(TMPTILEOUTFORMAT==FLOAT_DATA && outfiles->outfileformat!=FLOAT_DATA){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Precompiled tile format precludes given output format");
    }
    if(params->scndryarcflowmax<1){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Parameter scndryarcflowmax too small");
    }
    if(params->initonly){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Initialize-only mode and tile mode are mutually exclusive");
    }
    if(params->assembleonly){
      n=strlen(params->tiledir);
      while(--n>0 && params->tiledir[n]=='/'){
        params->tiledir[n]='\0';
      }
      if(!strlen(params->tiledir)){
        fflush(NULL);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Tile directory name must be specified");
      }
      if(!strcmp(params->tiledir,"/")){
        StrNCopy(params->tiledir,"",MAXSTRLEN);
      }
      params->rmtmptile=FALSE;     /* cowardly avoid removing tile dir input */
    }
    if(params->piecefirstrow!=DEF_PIECEFIRSTROW 
       || params->piecefirstcol!=DEF_PIECEFIRSTCOL
       || params->piecenrow!=DEF_PIECENROW
       || params->piecencol!=DEF_PIECENCOL){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Piece-only mode cannot be used with multiple tiles");
    }
    if(params->costmode==NOSTATCOSTS){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "no-statistical-costs option cannot be used in tile mode");
    }
    if(params->rowovrlp<TILEOVRLPWARNTHRESH
       || params->colovrlp<TILEOVRLPWARNTHRESH){
      fflush(NULL);
      fprintf(sp0,"WARNING: Tile overlap is small (may give bad results)\n");
    }
  }else{
    if(params->assembleonly){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "assemble-only mode can only be used with multiple tiles");
    }
    if(params->nthreads>1){
      fflush(NULL);
      fprintf(sp0,"only one tile--disregarding multiprocessor option\n");
    }
    if(params->rowovrlp || params->colovrlp){
      fflush(NULL);
      fprintf(sp0,"only one tile--disregarding tile overlap values\n");
    }
    if(params->onetilereopt){
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Cannot do single-tile reoptimization without tiling params");
    }
  }
  if(params->nthreads<1){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Number of processors must be at least one");
  }else if(params->nthreads>MAXTHREADS){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Number of processors exceeds precomplied limit of " +
            std::to_string(MAXTHREADS));
  }

  /* piece params */
  params->piecefirstrow--;                   /* index from 0 instead of 1 */
  params->piecefirstcol--;                   /* index from 0 instead of 1 */
  if(!params->piecenrow){
    params->piecenrow=nlines;
  }
  if(!params->piecencol){
    params->piecencol=linelen;
  }
  if(params->piecefirstrow<0 || params->piecefirstcol<0 
     || params->piecenrow<1 || params->piecencol<1
     || params->piecefirstrow+params->piecenrow>nlines
     || params->piecefirstcol+params->piecencol>linelen){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Illegal values for piece of interferogram to unwrap");
  }

  /* connected component parameters */
  if(params->regrowconncomps){
    if(!strlen(outfiles->conncompfile)){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "No connected component output file specified");
    }      
    params->unwrapped=TRUE;
  }
  if(params->minconncompfrac<0 || params->minconncompfrac>1){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Illegal value for minimum connected component fraction");
  }
  if(params->maxncomps<=0){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Illegal value for maximum number of connected components");
  }
  if(params->maxncomps>UCHAR_MAX
     && params->conncompouttype==CONNCOMPOUTTYPEUCHAR){
    fflush(NULL);
    fprintf(sp0,"WARNING: clipping max num conn comps to fit uchar out type\n");
    params->maxncomps=UCHAR_MAX;
  }
  if(params->maxncomps>UINT_MAX
     && params->conncompouttype==CONNCOMPOUTTYPEUINT){
    fflush(NULL);
    fprintf(sp0,"WARNING: clipping max num conn comps to fit uint out type\n");
    params->maxncomps=UINT_MAX;
  }
  if(strlen(outfiles->conncompfile)){
    if(params->initonly){
      fflush(NULL);
      fprintf(sp0,"WARNING: connected component mask cannot be generated "
              "in initialize-only mode\n         mask will not be output\n");
      StrNCopy(outfiles->conncompfile,"",MAXSTRLEN);
    }
    if(params->costmode==NOSTATCOSTS){
      fflush(NULL);
      fprintf(sp0,"WARNING: connected component mask cannot be generated "
              "without statistical costs\n         mask will not be output\n");
      StrNCopy(outfiles->conncompfile,"",MAXSTRLEN);
    }
  }

  /* done */
  return(0);

}


/* function: ReadConfigFile()
 * --------------------------
 * Read in parameter values from a file, overriding existing parameters.
 */
int ReadConfigFile(const char *conffile, infileT *infiles, outfileT *outfiles,
                   long *linelenptr, paramT *params){

  int parsestatus;
  long nlines, nparams;
  char *ptr;
  char buf[MAXLINELEN]={};
  FILE *fp;

  
  /* open input config file */
  if(strlen(conffile)){
    if((fp=fopen(conffile,"r"))==NULL){

      /* abort if we were given a non-zero length name that is unreadable */
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Unable to read configuration file " + std::string(conffile));
    }
  }else{
    
    /* if we were given a zero-length name, just ignore it and go on */
    return(0);
  }

  /* read each line and convert the first two fields */
  nlines=0;
  nparams=0;
  while(TRUE){

    /* read a line from the file and store it in buffer buf */
    buf[0]='\0';
    ptr=fgets(buf,MAXLINELEN,fp);

    /* break when we read EOF without reading any text */
    if(ptr==NULL && !strlen(buf)){
      break;
    }
    nlines++;

    /* make sure we got the whole line */
    if(strlen(buf)>=MAXLINELEN-1){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Line " + std::to_string(nlines) + " in file " +
              std::string(conffile) + " exceeds maximum line length");
    }

    /* parse config line */
    parsestatus=ParseConfigLine(buf,conffile,nlines,
                                infiles,outfiles,linelenptr,params);
    if(parsestatus>0){
      nparams++;
    }
    
  }

  /* finish up */
  fclose(fp);
  if(nparams>1){
    fprintf(sp1,"%ld parameters input from file %s (%ld lines total)\n",
            nparams,conffile,nlines);
  }else{
    if(nlines>1){
      fprintf(sp1,"%ld parameter input from file %s (%ld lines total)\n",
              nparams,conffile,nlines);
    }else{
      fprintf(sp1,"%ld parameter input from file %s (%ld line total)\n",
              nparams,conffile,nlines);
    }
  }

  /* done */
  return(0);

}


/* function: ParseConfigLine()
 * ---------------------------
 * Parse config line from passed buffer.
 */
static
int ParseConfigLine(char *buf, const char *conffile, long nlines,
                    infileT *infiles, outfileT *outfiles,
                    long *linelenptr, paramT *params){

  int nparams;
  long nfields;
  char str1[MAXLINELEN]={}, str2[MAXLINELEN]={};
  signed char badparam;

  /* set up */
  nparams=0;
  badparam=FALSE;
  
  /* read the first two fields */
  /* (str1, str2 same size as buf, so can't overflow them */
  nfields=sscanf(buf,"%s %s",str1,str2);

  /* if only one field is read, and it is not a comment, we have an error */
  if(nfields==1 && isalnum(str1[0])){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Unrecognized configuration parameter '" + std::string(str1) +
            "' (" + std::string(conffile) + ":" + std::to_string(nlines) + ")");
  }

  /* if we have (at least) two non-comment fields */
  if(nfields==2 && isalnum(str1[0])){

    /* do the conversions */
    nparams++;
    if(!strcmp(str1,"INFILE")){
      StrNCopy(infiles->infile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"OUTFILE")){
      StrNCopy(outfiles->outfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"WEIGHTFILE")){
      StrNCopy(infiles->weightfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"AMPFILE") || !strcmp(str1,"AMPFILE1")){
      if(strlen(infiles->ampfile2) && !params->amplitude){
        fflush(NULL);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Cannot specify both amplitude and power");
      }
      StrNCopy(infiles->ampfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"AMPFILE2")){
      if(strlen(infiles->ampfile) && !params->amplitude){
        fflush(NULL);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Cannot specify both amplitude and power");
      }
      StrNCopy(infiles->ampfile2,str2,MAXSTRLEN);
      infiles->ampfileformat=FLOAT_DATA;
    }else if(!strcmp(str1,"PWRFILE") || !strcmp(str1,"PWRFILE1")){
      if(strlen(infiles->ampfile2) && params->amplitude){
        fflush(NULL);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Cannot specify both amplitude and power");
      } 
      StrNCopy(infiles->ampfile,str2,MAXSTRLEN);
      params->amplitude=FALSE;
    }else if(!strcmp(str1,"PWRFILE2")){
      if(strlen(infiles->ampfile) && params->amplitude){
        fflush(NULL);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Cannot specify both amplitude and power");
      } 
      StrNCopy(infiles->ampfile2,str2,MAXSTRLEN);
      params->amplitude=FALSE;
      infiles->ampfileformat=FLOAT_DATA;
    }else if(!strcmp(str1,"MAGFILE")){
      StrNCopy(infiles->magfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"CORRFILE")){
      StrNCopy(infiles->corrfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"ESTIMATEFILE")){
      StrNCopy(infiles->estfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"LINELENGTH") || !strcmp(str1,"LINELEN")){
      badparam=StringToLong(str2,linelenptr);
    }else if(!strcmp(str1,"STATCOSTMODE")){
      if(!strcmp(str2,"TOPO")){
        params->costmode=TOPO;
      }else if(!strcmp(str2,"DEFO")){
        params->costmode=DEFO;
      }else if(!strcmp(str2,"SMOOTH")){
        params->costmode=SMOOTH;
      }else if(!strcmp(str2,"NOSTATCOSTS")){
        params->costmode=NOSTATCOSTS;
      }else{
        badparam=TRUE;
      }
    }else if(!strcmp(str1,"INITONLY")){
      badparam=SetBooleanSignedChar(&(params->initonly),str2);
    }else if(!strcmp(str1,"UNWRAPPED_IN")){
      badparam=SetBooleanSignedChar(&(params->unwrapped),str2);
    }else if(!strcmp(str1,"DEBUG") || !strcmp(str1,"DUMPALL")){
      badparam=SetBooleanSignedChar(&(params->dumpall),str2);
    }else if(!strcmp(str1,"VERBOSE")){
      badparam=SetBooleanSignedChar(&(params->verbose),str2);
    }else if(!strcmp(str1,"INITMETHOD")){
      if(!strcmp(str2,"MST") || !strcmp(str2,"mst")){
        params->initmethod=MSTINIT;
      }else if(!strcmp(str2,"MCF") || !strcmp(str2,"mcf")){
        params->initmethod=MCFINIT;
      }else{
        badparam=TRUE;
      }
    }else if(!strcmp(str1,"ORBITRADIUS")){
      if(!(badparam=StringToDouble(str2,&(params->orbitradius)))){
        params->altitude=0;
      }
    }else if(!strcmp(str1,"ALTITUDE")){
      if(!(badparam=StringToDouble(str2,&(params->altitude)))){
        params->orbitradius=0;
      }
    }else if(!strcmp(str1,"EARTHRADIUS")){
      badparam=StringToDouble(str2,&(params->earthradius));
    }else if(!strcmp(str1,"BPERP")){
      badparam=StringToDouble(str2,&(params->bperp));
    }else if(!strcmp(str1,"TRANSMITMODE")){
      if(!strcmp(str2,"PINGPONG") || !strcmp(str2,"REPEATPASS")){
        params->transmitmode=PINGPONG;
      }else if(!strcmp(str2,"SINGLEANTENNATRANSMIT") || !strcmp(str2,"SAT")
               || !strcmp(str2,"SINGLEANTTRANSMIT")){
        params->transmitmode=SINGLEANTTRANSMIT;
      }else{
        badparam=TRUE;
      }
    }else if(!strcmp(str1,"BASELINE")){
      if(!(badparam=StringToDouble(str2,&(params->baseline)))){
        params->bperp=0;
      }
    }else if(!strcmp(str1,"BASELINEANGLE_RAD")){
      if(!(badparam=StringToDouble(str2,&(params->baselineangle)))){
        params->bperp=0;
      }
    }else if(!strcmp(str1,"BASELINEANGLE_DEG")){
      if(!(badparam=StringToDouble(str2,&(params->baselineangle)))){
        (params->baselineangle)*=(PI/180.0);
        params->bperp=0;
      }
    }else if(!strcmp(str1,"NLOOKSRANGE")){
      badparam=StringToLong(str2,&(params->nlooksrange));
    }else if(!strcmp(str1,"NLOOKSAZ")){
      badparam=StringToLong(str2,&(params->nlooksaz));
    }else if(!strcmp(str1,"NLOOKSOTHER")){
      badparam=StringToLong(str2,&(params->nlooksother));
    }else if(!strcmp(str1,"NCORRLOOKS")){
      badparam=StringToDouble(str2,&(params->ncorrlooks));
    }else if(!strcmp(str1,"NCORRLOOKSRANGE")){
      badparam=StringToLong(str2,&(params->ncorrlooksrange));
    }else if(!strcmp(str1,"NCORRLOOKSAZ")){
      badparam=StringToLong(str2,&(params->ncorrlooksaz));
    }else if(!strcmp(str1,"NEARRANGE") || !strcmp(str1,"NOMRANGE")){
      badparam=StringToDouble(str2,&(params->nearrange));
    }else if(!strcmp(str1,"DR")){
      badparam=StringToDouble(str2,&(params->dr));
    }else if(!strcmp(str1,"DA")){
      badparam=StringToDouble(str2,&(params->da));
    }else if(!strcmp(str1,"RANGERES")){
      badparam=StringToDouble(str2,&(params->rangeres));
    }else if(!strcmp(str1,"AZRES")){
      badparam=StringToDouble(str2,&(params->azres));
    }else if(!strcmp(str1,"LAMBDA")){
      badparam=StringToDouble(str2,&(params->lambda));
    }else if(!strcmp(str1,"KDS") || !strcmp(str1,"KSD")){
      if(!strcmp(str1,"KSD")){
        fflush(NULL);
        fprintf(sp0,"WARNING: parameter KSD interpreted as KDS (%s:%ld)\n",
                conffile,nlines);
      }
      badparam=StringToDouble(str2,&(params->kds));
    }else if(!strcmp(str1,"SPECULAREXP") || !strcmp(str1,"N")){
      badparam=StringToDouble(str2,&(params->specularexp));
    }else if(!strcmp(str1,"DZRCRITFACTOR")){
      badparam=StringToDouble(str2,&(params->dzrcritfactor));
    }else if(!strcmp(str1,"SHADOW")){
      badparam=SetBooleanSignedChar(&(params->shadow),str2);
    }else if(!strcmp(str1,"DZEIMIN")){
      badparam=StringToDouble(str2,&(params->dzeimin));
    }else if(!strcmp(str1,"LAYWIDTH")){
      badparam=StringToLong(str2,&(params->laywidth));
    }else if(!strcmp(str1,"LAYMINEI")){
      badparam=StringToDouble(str2,&(params->layminei));
    }else if(!strcmp(str1,"SLOPERATIOFACTOR")){
      badparam=StringToDouble(str2,&(params->sloperatiofactor));
    }else if(!strcmp(str1,"SIGSQEI")){
      badparam=StringToDouble(str2,&(params->sigsqei));
    }else if(!strcmp(str1,"DRHO")){
      badparam=StringToDouble(str2,&(params->drho));
    }else if(!strcmp(str1,"RHOSCONST1")){
      badparam=StringToDouble(str2,&(params->rhosconst1));
    }else if(!strcmp(str1,"RHOSCONST2")){
      badparam=StringToDouble(str2,&(params->rhosconst2));
    }else if(!strcmp(str1,"CSTD1")){
      badparam=StringToDouble(str2,&(params->cstd1));
    }else if(!strcmp(str1,"CSTD2")){
      badparam=StringToDouble(str2,&(params->cstd2));
    }else if(!strcmp(str1,"CSTD3")){
      badparam=StringToDouble(str2,&(params->cstd3));
    }else if(!strcmp(str1,"DEFAULTCORR")){
      badparam=StringToDouble(str2,&(params->defaultcorr));
    }else if(!strcmp(str1,"RHOMINFACTOR")){
      badparam=StringToDouble(str2,&(params->rhominfactor));
    }else if(!strcmp(str1,"DZLAYPEAK")){
      badparam=StringToDouble(str2,&(params->dzlaypeak));
    }else if(!strcmp(str1,"AZDZFACTOR")){
      badparam=StringToDouble(str2,&(params->azdzfactor));
    }else if(!strcmp(str1,"DZEIFACTOR")){
      badparam=StringToDouble(str2,&(params->dzeifactor));
    }else if(!strcmp(str1,"DZEIWEIGHT")){
      badparam=StringToDouble(str2,&(params->dzeiweight));
    }else if(!strcmp(str1,"DZLAYFACTOR")){
      badparam=StringToDouble(str2,&(params->dzlayfactor));
    }else if(!strcmp(str1,"LAYCONST")){
      badparam=StringToDouble(str2,&(params->layconst));
    }else if(!strcmp(str1,"LAYFALLOFFCONST")){
      badparam=StringToDouble(str2,&(params->layfalloffconst));
    }else if(!strcmp(str1,"SIGSQSHORTMIN")){
      badparam=StringToLong(str2,&(params->sigsqshortmin));
    }else if(!strcmp(str1,"SIGSQLAYFACTOR")){
      badparam=StringToDouble(str2,&(params->sigsqlayfactor));
    }else if(!strcmp(str1,"DEFOAZDZFACTOR")){
      badparam=StringToDouble(str2,&(params->defoazdzfactor));
    }else if(!strcmp(str1,"DEFOTHRESHFACTOR")){
      badparam=StringToDouble(str2,&(params->defothreshfactor));
    }else if(!strcmp(str1,"DEFOMAX_CYCLE")){
      badparam=StringToDouble(str2,&(params->defomax));
    }else if(!strcmp(str1,"DEFOMAX_RAD")){
      if(!(badparam=StringToDouble(str2,&(params->defomax)))){
        params->defomax/=TWOPI;
      }
    }else if(!strcmp(str1,"SIGSQCORR")){
      badparam=StringToDouble(str2,&(params->sigsqcorr));
    }else if(!strcmp(str1,"DEFOLAYCONST") || !strcmp(str1,"DEFOCONST")){
      badparam=StringToDouble(str2,&(params->defolayconst));
    }else if(!strcmp(str1,"INITMAXFLOW")){
      badparam=StringToLong(str2,&(params->initmaxflow));
    }else if(!strcmp(str1,"ARCMAXFLOWCONST")){
      badparam=StringToLong(str2,&(params->arcmaxflowconst));
    }else if(!strcmp(str1,"MAXFLOW")){
      badparam=StringToLong(str2,&(params->maxflow));
    }else if(!strcmp(str1,"KROWEI") || !strcmp(str1,"KROW")){
      badparam=StringToLong(str2,&(params->krowei));
    }else if(!strcmp(str1,"KCOLEI") || !strcmp(str1,"KCOL")){
      badparam=StringToLong(str2,&(params->kcolei));
    }else if(!strcmp(str1,"KPERPDPSI")){
      badparam=StringToLong(str2,&(params->kperpdpsi));
    }else if(!strcmp(str1,"KPARDPSI")){
      badparam=StringToLong(str2,&(params->kpardpsi));
    }else if(!strcmp(str1,"THRESHOLD")){
      badparam=StringToDouble(str2,&(params->threshold));
    }else if(!strcmp(str1,"INITDZR")){
      badparam=StringToDouble(str2,&(params->initdzr));
    }else if(!strcmp(str1,"INITDZSTEP")){
      badparam=StringToDouble(str2,&(params->initdzstep));
    }else if(!strcmp(str1,"MAXCOST")){
      badparam=StringToDouble(str2,&(params->maxcost));
    }else if(!strcmp(str1,"COSTSCALE")){
      badparam=StringToDouble(str2,&(params->costscale));
    }else if(!strcmp(str1,"COSTSCALEAMBIGHT")){
      badparam=StringToDouble(str2,&(params->costscaleambight));
    }else if(!strcmp(str1,"DNOMINCANGLE")){
      badparam=StringToDouble(str2,&(params->dnomincangle));
    }else if(!strcmp(str1,"NMAJORPRUNE")){
      badparam=StringToLong(str2,&(params->nmajorprune));
    }else if(!strcmp(str1,"PRUNECOSTTHRESH")){
      badparam=StringToLong(str2,&(params->prunecostthresh));
    }else if(!strcmp(str1,"PLPN")){
      badparam=StringToDouble(str2,&(params->p));
    }else if(!strcmp(str1,"BIDIRLPN")){
      badparam=SetBooleanSignedChar(&(params->bidirlpn),str2);
    }else if(!strcmp(str1,"EDGEMASKTOP")){
      badparam=StringToLong(str2,&(params->edgemasktop));
    }else if(!strcmp(str1,"EDGEMASKBOT")){
      badparam=StringToLong(str2,&(params->edgemaskbot));
    }else if(!strcmp(str1,"EDGEMASKLEFT")){
      badparam=StringToLong(str2,&(params->edgemaskleft));
    }else if(!strcmp(str1,"EDGEMASKRIGHT")){
      badparam=StringToLong(str2,&(params->edgemaskright));
    }else if(!strcmp(str1,"PIECEFIRSTROW")){
      badparam=StringToLong(str2,&(params->piecefirstrow));
    }else if(!strcmp(str1,"PIECEFIRSTCOL")){
      badparam=StringToLong(str2,&(params->piecefirstcol));
    }else if(!strcmp(str1,"PIECENROW")){
      badparam=StringToLong(str2,&(params->piecenrow));
    }else if(!strcmp(str1,"PIECENCOL")){
      badparam=StringToLong(str2,&(params->piecencol));
    }else if(!strcmp(str1,"NTILEROW")){
      badparam=StringToLong(str2,&(params->ntilerow));
    }else if(!strcmp(str1,"NTILECOL")){
      badparam=StringToLong(str2,&(params->ntilecol));
    }else if(!strcmp(str1,"ROWOVRLP")){
      badparam=StringToLong(str2,&(params->rowovrlp));
    }else if(!strcmp(str1,"COLOVRLP")){
      badparam=StringToLong(str2,&(params->colovrlp));
    }else if(!strcmp(str1,"TILECOSTTHRESH")){
      badparam=StringToLong(str2,&(params->tilecostthresh));
    }else if(!strcmp(str1,"MINREGIONSIZE")){
      badparam=StringToLong(str2,&(params->minregionsize));
    }else if(!strcmp(str1,"TILEEDGEWEIGHT")){
      badparam=StringToDouble(str2,&(params->tileedgeweight));
    }else if(!strcmp(str1,"SCNDRYARCFLOWMAX")){
      badparam=StringToLong(str2,&(params->scndryarcflowmax));  
    }else if(!strcmp(str1,"TILEDIR")){
      StrNCopy(params->tiledir,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"ASSEMBLEONLY")){
      badparam=SetBooleanSignedChar(&(params->assembleonly),str2);
    }else if(!strcmp(str1,"SINGLETILEREOPTIMIZE")){
      badparam=SetBooleanSignedChar(&(params->onetilereopt),str2);
    }else if(!strcmp(str1,"RMTMPTILE")){
      badparam=SetBooleanSignedChar(&(params->rmtmptile),str2);
      params->rmtileinit=params->rmtmptile;
    }else if(!strcmp(str1,"MINCONNCOMPFRAC")){
      badparam=StringToDouble(str2,&(params->minconncompfrac));
    }else if(!strcmp(str1,"CONNCOMPTHRESH")){
      badparam=StringToLong(str2,&(params->conncompthresh));
    }else if(!strcmp(str1,"MAXNCOMPS")){
      badparam=StringToLong(str2,&(params->maxncomps));
    }else if(!strcmp(str1,"CONNCOMPOUTTYPE")){
      if(!strcmp(str2,"UCHAR")){
        params->conncompouttype=CONNCOMPOUTTYPEUCHAR;
      }else if(!strcmp(str2,"UINT")){
        params->conncompouttype=CONNCOMPOUTTYPEUINT;
      }else{
        badparam=TRUE;
      }
    }else if(!strcmp(str1,"NSHORTCYCLE")){
      badparam=StringToLong(str2,&(params->nshortcycle));
    }else if(!strcmp(str1,"MAXNEWNODECONST")){
      badparam=StringToDouble(str2,&(params->maxnewnodeconst));
    }else if(!strcmp(str1,"MAXNFLOWCYCLES")){
      badparam=StringToLong(str2,&(params->maxnflowcycles));
    }else if(!strcmp(str1,"MAXCYCLEFRACTION")){
      badparam=StringToDouble(str2,&(params->maxcyclefraction));
      params->maxnflowcycles=USEMAXCYCLEFRACTION;
    }else if(!strcmp(str1,"SOURCEMODE")){
      fflush(NULL);
      fprintf(sp0,
              "WARNING: SOURCEMODE keyword no longer supported--ignoring\n");
    }else if(!strcmp(str1,"NCONNNODEMIN")){
      badparam=StringToLong(str2,&(params->nconnnodemin));
    }else if(!strcmp(str1,"NPROC") || !strcmp(str1,"NTHREADS")){
      badparam=StringToLong(str2,&(params->nthreads));
    }else if(!strcmp(str1,"COSTINFILE")){
      StrNCopy(infiles->costinfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"BYTEMASKFILE")){
      StrNCopy(infiles->bytemaskfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"DOTILEMASKFILE")){
      StrNCopy(infiles->dotilemaskfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"COSTOUTFILE")){
      StrNCopy(outfiles->costoutfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"LOGFILE")){
      StrNCopy(outfiles->logfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"INFILEFORMAT")){
      if(!strcmp(str2,"COMPLEX_DATA")){
        infiles->infileformat=COMPLEX_DATA;
      }else if(!strcmp(str2,"FLOAT_DATA")){
        infiles->infileformat=FLOAT_DATA;
      }else if(!strcmp(str2,"ALT_LINE_DATA")){
        infiles->infileformat=ALT_LINE_DATA;
      }else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
        infiles->infileformat=ALT_SAMPLE_DATA;
      }else{
        badparam=TRUE;
      }
    }else if(!strcmp(str1,"UNWRAPPEDINFILEFORMAT")){
      if(!strcmp(str2,"ALT_LINE_DATA")){
        infiles->unwrappedinfileformat=ALT_LINE_DATA;
      }else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
        infiles->unwrappedinfileformat=ALT_SAMPLE_DATA;
      }else if(!strcmp(str2,"FLOAT_DATA")){
        infiles->unwrappedinfileformat=FLOAT_DATA;
      }else{
        badparam=TRUE;
      }
    }else if(!strcmp(str1,"MAGFILEFORMAT")){
      if(!strcmp(str2,"ALT_LINE_DATA")){
        infiles->magfileformat=ALT_LINE_DATA;
      }else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
        infiles->magfileformat=ALT_SAMPLE_DATA;
      }else if(!strcmp(str2,"FLOAT_DATA")){
        infiles->magfileformat=FLOAT_DATA;
      }else if(!strcmp(str2,"COMPLEX_DATA")){
        infiles->magfileformat=COMPLEX_DATA;
      }else{
        badparam=TRUE;
      }
    }else if(!strcmp(str1,"OUTFILEFORMAT")){
      if(!strcmp(str2,"ALT_LINE_DATA")){
        outfiles->outfileformat=ALT_LINE_DATA;
      }else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
        outfiles->outfileformat=ALT_SAMPLE_DATA;
      }else if(!strcmp(str2,"FLOAT_DATA")){
        outfiles->outfileformat=FLOAT_DATA;
      }else{
        badparam=TRUE;
      }
    }else if(!strcmp(str1,"CORRFILEFORMAT")){
      if(!strcmp(str2,"ALT_LINE_DATA")){
        infiles->corrfileformat=ALT_LINE_DATA;
      }else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
        infiles->corrfileformat=ALT_SAMPLE_DATA;
      }else if(!strcmp(str2,"FLOAT_DATA")){
        infiles->corrfileformat=FLOAT_DATA;
      }else{
        badparam=TRUE;
      }
    }else if(!strcmp(str1,"AMPFILEFORMAT")){
      if(!strcmp(str2,"ALT_LINE_DATA")){
        infiles->ampfileformat=ALT_LINE_DATA;
      }else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
        infiles->ampfileformat=ALT_SAMPLE_DATA;
      }else if(!strcmp(str2,"FLOAT_DATA")){
        infiles->ampfileformat=FLOAT_DATA;
      }else{
        badparam=TRUE;
      }
    }else if(!strcmp(str1,"ESTFILEFORMAT")){
      if(!strcmp(str2,"ALT_LINE_DATA")){
        infiles->estfileformat=ALT_LINE_DATA;
      }else if(!strcmp(str2,"ALT_SAMPLE_DATA")){
        infiles->estfileformat=ALT_SAMPLE_DATA;
      }else if(!strcmp(str2,"FLOAT_DATA")){
        infiles->estfileformat=FLOAT_DATA;
      }else{
        badparam=TRUE;
      }
    }else if(!strcmp(str1,"INITFILE")){
      StrNCopy(outfiles->initfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"FLOWFILE")){
      StrNCopy(outfiles->flowfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"EIFILE")){
      StrNCopy(outfiles->eifile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"ROWCOSTFILE")){
      StrNCopy(outfiles->rowcostfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"COLCOSTFILE")){
      StrNCopy(outfiles->colcostfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"MSTROWCOSTFILE")){
      StrNCopy(outfiles->mstrowcostfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"MSTCOLCOSTFILE")){
      StrNCopy(outfiles->mstcolcostfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"MSTCOSTSFILE")){
      StrNCopy(outfiles->mstcostsfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"CORRDUMPFILE")){
      StrNCopy(outfiles->corrdumpfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"RAWCORRDUMPFILE")){
      StrNCopy(outfiles->rawcorrdumpfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"CONNCOMPFILE")){
      StrNCopy(outfiles->conncompfile,str2,MAXSTRLEN);
    }else if(!strcmp(str1,"REGROWCONNCOMPS")){
      badparam=SetBooleanSignedChar(&(params->regrowconncomps),str2);
    }else{
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Unrecognized configuration parameter '" + std::string(str1) +
              "' (" + std::string(conffile) + ":" + std::to_string(nlines) +
              ")");
    }
    
    /* give an error if we had trouble interpreting the line */
    if(badparam){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Illegal argument " + std::string(str2) + " for parameter " +
              std::string(str1) + " (" + std::string(conffile) + ":" +
              std::to_string(nlines) + ")");
    }
      
  }

  /* return number of parameters successfully parsed */
  return(nparams);
  
}


/* function: WriteConfigLogFile()
 * ------------------------------
 * Writes a text log file of configuration parameters and other
 * information.  The log file is in a format compatible to be used as
 * a configuration file.  
 */
int WriteConfigLogFile(infileT *infiles, 
                       outfileT *outfiles, long linelen, paramT *params){

  FILE *fp;
  time_t t[1]={};
  char buf[MAXSTRLEN]={}, *ptr;
  char hostnamestr[MAXSTRLEN]={};

  /* see if we need to write a log file */
  if(strlen(outfiles->logfile)){

    /* open the log file */
    if((fp=fopen(outfiles->logfile,"w"))==NULL){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Unable to write to log file " + std::string(outfiles->logfile));
    }
    fprintf(sp1,"Logging run-time parameters to file %s\n",outfiles->logfile);
    
    /* print some run-time environment information */
    fprintf(fp,"# %s v%s\n",PROGRAMNAME,VERSION);
    time(t);
    fprintf(fp,"# Log file generated %s",ctime(t));
    if(gethostname(hostnamestr,MAXSTRLEN)){
      fprintf(fp,"# Could not determine host name\n");
    }else{
      fprintf(fp,"# Host name: %s\n",hostnamestr);
    }
    fprintf(fp,"# PID %ld\n",params->parentpid);
    ptr=getcwd(buf,MAXSTRLEN);
    if(ptr!=NULL){
      fprintf(fp,"# Current working directory: %s\n",buf);
    }else{
      fprintf(fp,"# Could not determine current working directory\n");
    }
    fprintf(fp,"\n\n");

    /* print some information about data type sizes */
    fprintf(fp,"# Data type size information for executable as compiled\n");
    fprintf(fp,"# sizeof(short):      %ld\n",sizeof(short));
    fprintf(fp,"# sizeof(int):        %ld\n",sizeof(int));
    fprintf(fp,"# sizeof(long):       %ld\n",sizeof(long));
    fprintf(fp,"# sizeof(float):      %ld\n",sizeof(float));
    fprintf(fp,"# sizeof(double):     %ld\n",sizeof(double));
    fprintf(fp,"# sizeof(void *):     %ld\n",sizeof(void *));
    fprintf(fp,"# sizeof(size_t):     %ld\n",sizeof(size_t));
    fprintf(fp,"\n");

    /* print an entry for each run-time parameter */
    /* input and output files and main runtime options */
    fprintf(fp,"# File input and output and runtime options\n");
    LogStringParam(fp,"INFILE",infiles->infile);
    fprintf(fp,"LINELENGTH  %ld\n",linelen);
    LogStringParam(fp,"OUTFILE",outfiles->outfile);
    LogStringParam(fp,"WEIGHTFILE",infiles->weightfile);
    if(params->amplitude){
      if(strlen(infiles->ampfile2)){
        LogStringParam(fp,"AMPFILE1",infiles->ampfile);
        LogStringParam(fp,"AMPFILE2",infiles->ampfile2);
      }else{
        LogStringParam(fp,"AMPFILE",infiles->ampfile);
      }
    }else{
      if(strlen(infiles->ampfile2)){
        LogStringParam(fp,"PWRFILE1",infiles->ampfile);
        LogStringParam(fp,"PWRFILE2",infiles->ampfile2);
      }else{
        LogStringParam(fp,"PWRFILE",infiles->ampfile);
      }
    }
    LogStringParam(fp,"MAGFILE",infiles->magfile);
    LogStringParam(fp,"CORRFILE",infiles->corrfile);
    LogStringParam(fp,"ESTIMATEFILE",infiles->estfile);
    LogStringParam(fp,"COSTINFILE",infiles->costinfile);
    LogStringParam(fp,"COSTOUTFILE",outfiles->costoutfile);
    LogStringParam(fp,"BYTEMASKFILE",infiles->bytemaskfile);
    LogStringParam(fp,"LOGFILE",outfiles->logfile);
    if(params->costmode==TOPO){
      fprintf(fp,"STATCOSTMODE  TOPO\n");
    }else if(params->costmode==DEFO){
      fprintf(fp,"STATCOSTMODE  DEFO\n");
    }else if(params->costmode==SMOOTH){
      fprintf(fp,"STATCOSTMODE  SMOOTH\n");
    }else if(params->costmode==NOSTATCOSTS){
      fprintf(fp,"STATCOSTMODE  NOSTATCOSTS\n");
    }
    LogBoolParam(fp,"INITONLY",params->initonly);
    LogBoolParam(fp,"UNWRAPPED_IN",params->unwrapped);
    LogBoolParam(fp,"DEBUG",params->dumpall);
    if(params->initmethod==MSTINIT){
      fprintf(fp,"INITMETHOD  MST\n");
    }else if(params->initmethod==MCFINIT){
      fprintf(fp,"INITMETHOD  MCF\n");
    }
    LogBoolParam(fp,"VERBOSE",params->verbose);

    /* file formats */
    fprintf(fp,"\n# File Formats\n");
    LogFileFormat(fp,"INFILEFORMAT",infiles->infileformat);
    LogFileFormat(fp,"OUTFILEFORMAT",outfiles->outfileformat);
    LogFileFormat(fp,"AMPFILEFORMAT",infiles->ampfileformat);
    LogFileFormat(fp,"MAGFILEFORMAT",infiles->magfileformat);
    LogFileFormat(fp,"CORRFILEFORMAT",infiles->corrfileformat);
    LogFileFormat(fp,"ESTFILEFORMAT",infiles->estfileformat);
    LogFileFormat(fp,"UNWRAPPEDINFILEFORMAT",infiles->unwrappedinfileformat);

    /* SAR and geometry parameters */
    fprintf(fp,"\n# SAR and Geometry Parameters\n");
    fprintf(fp,"ALTITUDE  %.8f\n",
            params->orbitradius-params->earthradius);
    fprintf(fp,"# ORBITRADIUS  %.8f\n",params->orbitradius);
    fprintf(fp,"EARTHRADIUS  %.8f\n",params->earthradius);
    if(params->bperp){
      fprintf(fp,"BPERP  %.8f\n",params->bperp);
    }else{
      fprintf(fp,"BASELINE %.8f\n",params->baseline);
      fprintf(fp,"BASELINEANGLE_DEG %.8f\n",
              params->baselineangle*(180.0/PI));
    }
    if(params->transmitmode==PINGPONG){
      fprintf(fp,"TRANSMITMODE  REPEATPASS\n");
    }else if(params->transmitmode==SINGLEANTTRANSMIT){
      fprintf(fp,"TRANSMITMODE  SINGLEANTENNATRANSMIT\n");
    }
    fprintf(fp,"NEARRANGE  %.8f\n",params->nearrange);
    fprintf(fp,"DR  %.8f\n",params->dr);
    fprintf(fp,"DA  %.8f\n",params->da);
    fprintf(fp,"RANGERES  %.8f\n",params->rangeres);
    fprintf(fp,"AZRES  %.8f\n",params->azres);
    fprintf(fp,"LAMBDA  %.8f\n",params->lambda);
    fprintf(fp,"NLOOKSRANGE  %ld\n",params->nlooksrange);
    fprintf(fp,"NLOOKSAZ  %ld\n",params->nlooksaz);
    fprintf(fp,"NLOOKSOTHER  %ld\n",params->nlooksother);
    fprintf(fp,"NCORRLOOKS  %.8f\n",params->ncorrlooks);
    fprintf(fp,"NCORRLOOKSRANGE  %ld\n",params->ncorrlooksrange);
    fprintf(fp,"NCORRLOOKSAZ  %ld\n",params->ncorrlooksaz);
      
    /* scattering model parameters */
    fprintf(fp,"\n# Scattering model parameters\n");
    fprintf(fp,"KDS  %.8f\n",params->kds);
    fprintf(fp,"SPECULAREXP  %.8f\n",params->specularexp);
    fprintf(fp,"DZRCRITFACTOR  %.8f\n",params->dzrcritfactor);
    LogBoolParam(fp,"SHADOW",params->shadow);
    fprintf(fp,"DZEIMIN  %.8f\n",params->dzeimin);
    fprintf(fp,"LAYWIDTH  %ld\n",params->laywidth);
    fprintf(fp,"LAYMINEI  %.8f\n",params->layminei);
    fprintf(fp,"SLOPERATIOFACTOR  %.8f\n",params->sloperatiofactor);
    fprintf(fp,"SIGSQEI  %.8f\n",params->sigsqei);
    
    /* decorrelation model paramters */
    fprintf(fp,"\n# Decorrelation model parameters\n");
    fprintf(fp,"DRHO  %.8f\n",params->drho);
    fprintf(fp,"RHOSCONST1  %.8f\n",params->rhosconst1);
    fprintf(fp,"RHOSCONST2  %.8f\n",params->rhosconst2);
    fprintf(fp,"CSTD1  %.8f\n",params->cstd1);
    fprintf(fp,"CSTD2  %.8f\n",params->cstd2);
    fprintf(fp,"CSTD3  %.8f\n",params->cstd3);
    fprintf(fp,"DEFAULTCORR  %.8f\n",params->defaultcorr);
    fprintf(fp,"RHOMINFACTOR  %.8f\n",params->rhominfactor);
      
    /* PDF model paramters */
    fprintf(fp,"\n# PDF model parameters\n");
    fprintf(fp,"DZLAYPEAK  %.8f\n",params->dzlaypeak);
    fprintf(fp,"AZDZFACTOR  %.8f\n",params->azdzfactor);
    fprintf(fp,"DZEIFACTOR  %.8f\n",params->dzeifactor);
    fprintf(fp,"DZEIWEIGHT  %.8f\n",params->dzeiweight);
    fprintf(fp,"DZLAYFACTOR  %.8f\n",params->dzlayfactor);
    fprintf(fp,"LAYCONST  %.8f\n",params->layconst);
    fprintf(fp,"LAYFALLOFFCONST  %.8f\n",params->layfalloffconst);
    fprintf(fp,"SIGSQSHORTMIN  %ld\n",params->sigsqshortmin);
    fprintf(fp,"SIGSQLAYFACTOR  %.8f\n",params->sigsqlayfactor);

    /* deformation mode paramters */
    fprintf(fp,"\n# Deformation mode parameters\n");
    fprintf(fp,"DEFOAZDZFACTOR  %.8f\n",params->defoazdzfactor);
    fprintf(fp,"DEFOTHRESHFACTOR  %.8f\n",params->defothreshfactor);
    fprintf(fp,"DEFOMAX_CYCLE  %.8f\n",params->defomax);
    fprintf(fp,"SIGSQCORR  %.8f\n",params->sigsqcorr);
    fprintf(fp,"DEFOCONST  %.8f\n",params->defolayconst);

    /* algorithm parameters */
    fprintf(fp,"\n# Algorithm parameters\n");
    fprintf(fp,"INITMAXFLOW  %ld\n",params->initmaxflow);
    fprintf(fp,"ARCMAXFLOWCONST  %ld\n",params->arcmaxflowconst);
    fprintf(fp,"MAXFLOW  %ld\n",params->maxflow);
    fprintf(fp,"KROWEI  %ld\n",params->krowei);
    fprintf(fp,"KCOLEI  %ld\n",params->kcolei);
    fprintf(fp,"KPARDPSI  %ld\n",params->kpardpsi);
    fprintf(fp,"KPERPDPSI  %ld\n",params->kperpdpsi);
    fprintf(fp,"THRESHOLD  %.8f\n",params->threshold);
    fprintf(fp,"INITDZR  %.8f\n",params->initdzr);
    fprintf(fp,"INITDZSTEP  %.8f\n",params->initdzstep);
    fprintf(fp,"MAXCOST  %.8f\n",params->maxcost);
    fprintf(fp,"COSTSCALE  %.8f\n",params->costscale);
    fprintf(fp,"COSTSCALEAMBIGHT  %.8f\n",params->costscaleambight);
    fprintf(fp,"DNOMINCANGLE  %.8f\n",params->dnomincangle);
    fprintf(fp,"NSHORTCYCLE  %ld\n",params->nshortcycle);
    fprintf(fp,"MAXNEWNODECONST  %.8f\n",params->maxnewnodeconst);
    if(params->maxnflowcycles==USEMAXCYCLEFRACTION){
      fprintf(fp,"MAXCYCLEFRACTION  %.8f\n",params->maxcyclefraction);
    }else{
      fprintf(fp,"MAXNFLOWCYCLES  %ld\n",params->maxnflowcycles);
    }
    fprintf(fp,"NCONNNODEMIN  %ld\n",params->nconnnodemin);
    fprintf(fp,"NMAJORPRUNE  %ld\n",params->nmajorprune);
    fprintf(fp,"PRUNECOSTTHRESH  %ld\n",params->prunecostthresh);
    if(params->p!=PROBCOSTP){
      fprintf(fp,"PLPN  %.8g\n",params->p);
      LogBoolParam(fp,"BIDIRLPN",params->bidirlpn);
    }else{
      fprintf(fp,"# PLPN  %.8g  (not set)\n",params->p);
      LogBoolParam(fp,"# BIDIRLPN",params->bidirlpn);
    }
      
    /* file names for dumping intermediate arrays */
    fprintf(fp,"\n# File names for dumping intermediate arrays\n");
    LogStringParam(fp,"INITFILE",outfiles->initfile);
    LogStringParam(fp,"FLOWFILE",outfiles->flowfile);
    LogStringParam(fp,"EIFILE",outfiles->eifile);
    LogStringParam(fp,"ROWCOSTFILE",outfiles->rowcostfile);
    LogStringParam(fp,"COLCOSTFILE",outfiles->colcostfile);
    LogStringParam(fp,"MSTROWCOSTFILE",outfiles->mstrowcostfile);
    LogStringParam(fp,"MSTCOLCOSTFILE",outfiles->mstcolcostfile);
    LogStringParam(fp,"MSTCOSTSFILE",outfiles->mstcostsfile);
    LogStringParam(fp,"RAWCORRDUMPFILE",outfiles->rawcorrdumpfile);
    LogStringParam(fp,"CORRDUMPFILE",outfiles->corrdumpfile);

    /* edge masking parameters */
    fprintf(fp,"\n# Edge masking parameters\n");
    fprintf(fp,"EDGEMASKTOP    %ld\n",params->edgemasktop);
    fprintf(fp,"EDGEMASKBOT    %ld\n",params->edgemaskbot);
    fprintf(fp,"EDGEMASKLEFT   %ld\n",params->edgemaskleft);
    fprintf(fp,"EDGEMASKRIGHT  %ld\n",params->edgemaskright);

    /* piece extraction parameters */
    if(params->ntilerow==1 && params->ntilecol==1){
      fprintf(fp,"\n# Piece extraction parameters\n");
      fprintf(fp,"PIECEFIRSTROW  %ld\n",params->piecefirstrow+1);
      fprintf(fp,"PIECEFIRSTCOL  %ld\n",params->piecefirstcol+1);
      fprintf(fp,"PIECENROW  %ld\n",params->piecenrow);
      fprintf(fp,"PIECENCOL  %ld\n",params->piecencol);
    }else{
      fprintf(fp,"\n# Piece extraction parameters\n");
      fprintf(fp,"# Parameters ignored because of tile mode\n");
      fprintf(fp,"# PIECEFIRSTROW  %ld\n",params->piecefirstrow);
      fprintf(fp,"# PIECEFIRSTCOL  %ld\n",params->piecefirstcol);
      fprintf(fp,"# PIECENROW  %ld\n",params->piecenrow);
      fprintf(fp,"# PIECENCOL  %ld\n",params->piecencol);
    }

    /* tile control */
    fprintf(fp,"\n# Tile control\n");
    fprintf(fp,"NTILEROW  %ld\n",params->ntilerow);
    fprintf(fp,"NTILECOL  %ld\n",params->ntilecol);
    fprintf(fp,"ROWOVRLP  %ld\n",params->rowovrlp);
    fprintf(fp,"COLOVRLP  %ld\n",params->colovrlp);
    fprintf(fp,"NPROC  %ld\n",params->nthreads);
    fprintf(fp,"TILECOSTTHRESH  %ld\n",params->tilecostthresh);
    fprintf(fp,"MINREGIONSIZE  %ld\n",params->minregionsize);
    fprintf(fp,"TILEEDGEWEIGHT  %.8f\n",params->tileedgeweight);
    fprintf(fp,"SCNDRYARCFLOWMAX  %ld\n",params->scndryarcflowmax);
    LogBoolParam(fp,"RMTMPTILE",params->rmtmptile);
    LogStringParam(fp,"DOTILEMASKFILE",infiles->dotilemaskfile);
    LogStringParam(fp,"TILEDIR",params->tiledir);
    LogBoolParam(fp,"ASSEMBLEONLY",params->assembleonly);
    LogBoolParam(fp,"SINGLETILEREOPTIMIZE",params->onetilereopt);

    /* connected component control */
    fprintf(fp,"\n# Connected component control\n");
    LogStringParam(fp,"CONNCOMPFILE",outfiles->conncompfile);
    LogBoolParam(fp,"REGROWCONNCOMPS",params->regrowconncomps);
    fprintf(fp,"MINCONNCOMPFRAC  %.8f\n",params->minconncompfrac);
    fprintf(fp,"CONNCOMPTHRESH  %ld\n",params->conncompthresh);
    fprintf(fp,"MAXNCOMPS  %ld\n",params->maxncomps);
    if(params->conncompouttype==CONNCOMPOUTTYPEUCHAR){
      fprintf(fp,"CONNCOMPOUTTYPE  UCHAR\n");
    }else if(params->conncompouttype==CONNCOMPOUTTYPEUINT){
      fprintf(fp,"CONNCOMPOUTTYPE  UINT\n");
    }else{
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Bad value of params->conncompouttype");
    }

    /* close the log file */
    if(fclose(fp)){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error in closing log file " + std::string(outfiles->logfile) +
              " (disk full?)");
    }
  }

  /* done */
  return(0);

}


/* function: LogStringParam()
 * --------------------------
 * Writes a line to the log file stream for the given keyword/value 
 * pair.
 */
static
int LogStringParam(FILE *fp, const char *key, char *value){

  /* see if we were passed a zero length value string */
  if(strlen(value)){
    fprintf(fp,"%s  %s\n",key,value);
    fflush(fp);
  }else{
    fprintf(fp,"# Empty value for parameter %s\n",key);
  }
  return(0);
}


/* LogBoolParam()
 * --------------
 * Writes a line to the log file stream for the given keyword/bool
 * pair.
 */
static
int LogBoolParam(FILE *fp, const char *key, signed char boolvalue){

  if(boolvalue){
    fprintf(fp,"%s  TRUE\n",key);
  }else{
    fprintf(fp,"%s  FALSE\n",key);
  }
  return(0);
}

/* LogFileFormat()
 * ---------------
 * Writes a line to the log file stream for the given keyword/
 * file format pair.
 */
static
int LogFileFormat(FILE *fp, const char *key, signed char fileformat){
  
  if(fileformat==COMPLEX_DATA){
    fprintf(fp,"%s  COMPLEX_DATA\n",key);
  }else if(fileformat==FLOAT_DATA){
    fprintf(fp,"%s  FLOAT_DATA\n",key);
  }else if(fileformat==ALT_LINE_DATA){
    fprintf(fp,"%s  ALT_LINE_DATA\n",key);
  }else if(fileformat==ALT_SAMPLE_DATA){
    fprintf(fp,"%s  ALT_SAMPLE_DATA\n",key);
  }
  return(0);
}


/* function: GetNLines() 
 * ---------------------
 * Gets the number of lines of data in the input file based on the file 
 * size.
 */
long GetNLines(infileT *infiles, long linelen, paramT *params){

  FILE *fp;
  long filesize, datasize;

  /* get size of input file in rows and columns */
  if((fp=fopen(infiles->infile,"r"))==NULL){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Can't open file " + std::string(infiles->infile));
  }
  fseek(fp,0,SEEK_END);
  filesize=ftell(fp);
  fclose(fp);
  if((!params->unwrapped && infiles->infileformat==FLOAT_DATA)
     || (params->unwrapped && infiles->unwrappedinfileformat==FLOAT_DATA)){
    datasize=sizeof(float);
  }else{
    datasize=2*sizeof(float);
  }
  if(filesize % (datasize*linelen)){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Extra data in file " + std::string(infiles->infile) +
            " (bad linelength?)");
  }
  return(filesize/(datasize*linelen));               /* implicit floor */

}


/* function: WriteOutputFile()
 * ---------------------------
 * Writes the unwrapped phase to the output file specified, in the
 * format given in the parameter structure.
 */
int WriteOutputFile(Array2D<float>& mag,
                    Array2D<float>& unwrappedphase, char *outfile,
                    outfileT *outfiles, long nrow, long ncol){

  if(outfiles->outfileformat==ALT_LINE_DATA){
    WriteAltLineFile(mag,unwrappedphase,outfile,nrow,ncol);
  }else if(outfiles->outfileformat==ALT_SAMPLE_DATA){
    WriteAltSampFile(mag,unwrappedphase,outfile,nrow,ncol);
  }else if(outfiles->outfileformat==FLOAT_DATA){
    Write2DArray<float>(unwrappedphase,outfile,
                        nrow,ncol,sizeof(float));
  }else{
    fflush(NULL);
    fprintf(sp0,"WARNING: Illegal format specified for output file\n");
    fprintf(sp0,"         using default floating-point format\n");
    Write2DArray<float>(unwrappedphase,outfile,
                        nrow,ncol,sizeof(float));
  }
  return(0);
}


/* function: OpenOutputFile()
 * --------------------------
 * Opens a file for writing.  If unable to open the file, tries to 
 * open a file in a dump path.  The name of the opened output file
 * is written into the string realoutfile, for which at least 
 * MAXSTRLEN bytes should already be allocated.
 */
FILE *OpenOutputFile(const char *outfile, char *realoutfile){

  char path[MAXSTRLEN]={}, basename[MAXSTRLEN]={}, dumpfile[MAXSTRLEN]={};
  FILE *fp;

  if((fp=fopen(outfile,"w"))==NULL){

    /* if we can't write to the out file, get the file name from the path */
    /* and dump to the default path */
    ParseFilename(outfile,path,basename);
    StrNCopy(dumpfile,DUMP_PATH,MAXSTRLEN);
    strcat(dumpfile,basename);
    if((fp=fopen(dumpfile,"w"))!=NULL){
      fflush(NULL);
      fprintf(sp0,"WARNING: Can't write to file %s.  Dumping to file %s\n",
             outfile,dumpfile);
      StrNCopy(realoutfile,dumpfile,MAXSTRLEN);
    }else{
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Unable to write to file " + std::string(outfile) + " or dump "
              "to file " + std::string(dumpfile));
    }
  }else{
    StrNCopy(realoutfile,outfile,MAXSTRLEN);
  }
  return(fp);

}


/* function: WriteAltLineFile()
 * ----------------------------
 * Writes magnitude and phase data from separate arrays to file.
 * Data type is float.  For each line of data, a full line of magnitude data
 * is written, then a full line of phase data.  Dumps the file to a 
 * default directory if the file name/path passed in cannot be used.
 */
static
int WriteAltLineFile(Array2D<float>& mag,
                     Array2D<float>& phase,
                     char *outfile, long nrow, long ncol){

  int row;
  FILE *fp;
  char realoutfile[MAXSTRLEN]={};

  fp=OpenOutputFile(outfile,realoutfile);
  for(row=0; row<nrow; row++){
    if(fwrite(mag.row(row).data(),sizeof(float),ncol,fp)!=ncol
       || fwrite(phase.row(row).data(),sizeof(float),ncol,fp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while writing to file " + std::string(realoutfile) +
              " (device full?)");
    }
  }
  if(fclose(fp)){
    fprintf(sp0,"WARNING: problem closing file %s (disk full?)\n",realoutfile);
  }
  return(0);
}


/* function: WriteAltSampFile()
 * ----------------------------
 * Writes data from separate arrays to file, alternating samples.
 * Data type is float.  nrow and ncol are the sizes of each input
 * array.  Dumps the file to a default directory if the file name/path 
 * passed in cannot be used.
 */
static
int WriteAltSampFile(Array2D<float>& arr1,
                     Array2D<float>& arr2,
                     char *outfile, long nrow, long ncol){

  long row, col;
  FILE *fp;
  char realoutfile[MAXSTRLEN]={};

  auto outline=Array1D<float>(2*ncol);
  fp=OpenOutputFile(outfile,realoutfile);
  for(row=0; row<nrow; row++){
    for(col=0;col<ncol;col++){
      outline[2*col]=arr1(row,col);
      outline[2*col+1]=arr2(row,col);
    }
    if(fwrite(outline.data(),sizeof(float),2*ncol,fp)!=2*ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while writing to file " + std::string(realoutfile) +
              " (device full?)");
    }
  }
  if(fclose(fp)){
    fflush(NULL);
    fprintf(sp0,"WARNING: problem closing file %s (disk full?)\n",realoutfile);
  }
  return(0);
}


/* function: Write2DArray()
 * ------------------------ 
 * Write data in a two dimensional array to a file.  Data elements are
 * have the number of bytes specified by size (use sizeof() when 
 * calling this function.  
 */
int Write2DArray(void **array, char *filename, long nrow, long ncol, 
                 size_t size){

  int row;
  FILE *fp;
  char realoutfile[MAXSTRLEN]={};

  fp=OpenOutputFile(filename,realoutfile);
  for(row=0; row<nrow; row++){
    if(fwrite(array[row],size,ncol,fp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while writing to file " + std::string(realoutfile) +
              " (device full?)");
    }
  }
  if(fclose(fp)){
    fflush(NULL);
    fprintf(sp0,"WARNING: problem closing file %s (disk full?)\n",realoutfile);
  }
  return(0);
}


/* function: Write2DRowColArray()
 * ------------------------------ 
 * Write data in a 2-D row-and-column array to a file.  Data elements 
 * have the number of bytes specified by size (use sizeof() when 
 * calling this function.  The format of the array is nrow-1 rows
 * of ncol elements, followed by nrow rows of ncol-1 elements each.
 */
int Write2DRowColArray(void **array, char *filename, long nrow, 
                        long ncol, size_t size){

  int row;
  FILE *fp;
  char realoutfile[MAXSTRLEN]={};

  fp=OpenOutputFile(filename,realoutfile);
  for(row=0; row<nrow-1; row++){
    if(fwrite(array[row],size,ncol,fp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while writing to file " + std::string(realoutfile) +
              " (device full?)");
    }
  }
  for(row=nrow-1; row<2*nrow-1; row++){
    if(fwrite(array[row],size,ncol-1,fp)!=ncol-1){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while writing to file " + std::string(realoutfile) +
              " (device full?)");
    }
  }
  if(fclose(fp)){
    fflush(NULL);
    fprintf(sp0,"WARNING: problem closing file %s (disk full?)\n",realoutfile);
  }
  return(0);
}


/* function: ReadInputFile()
 * -------------------------
 * Reads the input file specified on the command line.
 */
int ReadInputFile(infileT *infiles, Array2D<float>* magptr, Array2D<float>* wrappedphaseptr,
                  Array2D<short>* flowsptr, long linelen, long nlines,
                  paramT *params, tileparamT *tileparams){

  long row, col, nrow, ncol;

  /* initialize */
  Array2D<float> mag, wrappedphase, unwrappedphase;
  Array2D<short> flows;
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;

  /* check data size */
  if(tileparams->ncol>LARGESHORT || tileparams->nrow>LARGESHORT){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "One or more interferogram dimensions too large");
  }
  if(tileparams->ncol<2 || tileparams->nrow<2){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Input interferogram must be at least 2x2");
  }

  /* is the input file already unwrapped? */
  if(!params->unwrapped){

    /* read wrapped phase and possibly interferogram magnitude data */
    fprintf(sp1,"Reading wrapped phase from file %s\n",infiles->infile);
    if(infiles->infileformat==COMPLEX_DATA){
      ReadComplexFile(&mag,&wrappedphase,infiles->infile,
                      linelen,nlines,tileparams);
    }else if(infiles->infileformat==ALT_LINE_DATA){
      ReadAltLineFile(&mag,&wrappedphase,infiles->infile,
                      linelen,nlines,tileparams);
    }else if(infiles->infileformat==ALT_SAMPLE_DATA){
      ReadAltSampFile(&mag,&wrappedphase,infiles->infile,
                      linelen,nlines,tileparams);
    }else if(infiles->infileformat==FLOAT_DATA){
      Read2DArray(&wrappedphase,infiles->infile,linelen,nlines,
                  tileparams,sizeof(float *),sizeof(float));
    }else{
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Illegal input file format specification");
    }

    /* check to make sure the input data doesn't contain NaNs or infs */
    if(!ValidDataArray(wrappedphase,nrow,ncol)
       || (mag.size() && !ValidDataArray(mag,nrow,ncol))){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "NaN or infinity found in input float data");
    }
    if(mag.size() && !NonNegDataArray(mag,nrow,ncol)){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Negative magnitude found in input magnitude data");
    }

    /* flip the sign of the wrapped phase if flip flag is set */
    FlipPhaseArraySign(wrappedphase,params,nrow,ncol);

    /* make sure the wrapped phase is properly wrapped */
    WrapPhase(wrappedphase,nrow,ncol);

  }else{

    /* read unwrapped phase input */
    fprintf(sp1,"Reading unwrapped phase from file %s\n",infiles->infile);
    if(infiles->unwrappedinfileformat==ALT_LINE_DATA){
      ReadAltLineFile(&mag,&unwrappedphase,infiles->infile,
                      linelen,nlines,tileparams);
    }else if(infiles->unwrappedinfileformat==ALT_SAMPLE_DATA){
      ReadAltSampFile(&mag,&unwrappedphase,infiles->infile,
                           linelen,nlines,tileparams);
    }else if(infiles->unwrappedinfileformat==FLOAT_DATA){
      Read2DArray(&unwrappedphase,infiles->infile,linelen,nlines,
                  tileparams,sizeof(float *),sizeof(float));
    }else{
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Illegal input file format specification");
    }

    /* check to make sure the input data doesn't contain NaNs or infs */
    if(!ValidDataArray(unwrappedphase,nrow,ncol)
       || (mag.size() && !ValidDataArray(mag,nrow,ncol))){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "NaN or infinity found in input float data");
    }
    if(mag.size() && !NonNegDataArray(mag,nrow,ncol)){
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Negative magnitude found in input magnitude data");
    }

    
    /* flip the sign of the input unwrapped phase if flip flag is set */
    FlipPhaseArraySign(unwrappedphase,params,nrow,ncol);

    /* parse flows of unwrapped phase */
    wrappedphase=ExtractFlow(unwrappedphase,&flows,nrow,ncol);

  }

  /* show which pixels read if tiling */
  if(tileparams->nrow!=nlines || tileparams->ncol!=linelen){
    fprintf(sp2,
            "Read %ldx%ld array of pixels starting at row,col %ld,%ld\n",
            tileparams->nrow,tileparams->ncol,
            tileparams->firstrow,tileparams->firstcol);
  }

  /* get memory for mag (power) image and set to unity if not passed */
  if(!mag.size()) {
    mag=Array2D<float>(nrow, ncol);
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        mag(row,col)=1.0;
      }
    }
  }

  /* set passed pointers and return the number of rows in data */
  *wrappedphaseptr=wrappedphase;
  *magptr=mag;
  *flowsptr=flows;

  /* done */
  return(0);

}


/* function: ReadMagnitude()
 * -------------------------
 * Reads the interferogram magnitude in the specfied file if it exists.
 * Memory for the magnitude array should already have been allocated by
 * ReadInputFile().
 */
int ReadMagnitude(Array2D<float>& mag, infileT *infiles, long linelen, long nlines,
                  tileparamT *tileparams){

  Array2D<float> dummy;

  if(strlen(infiles->magfile)){
    fprintf(sp1,"Reading interferogram magnitude from file %s\n",
            infiles->magfile);
    if(infiles->magfileformat==FLOAT_DATA){
      Read2DArray(&mag,infiles->magfile,linelen,nlines,tileparams,
                  sizeof(float *),sizeof(float));
    }else if(infiles->magfileformat==COMPLEX_DATA){
      ReadComplexFile(&mag,&dummy,infiles->magfile,linelen,nlines,
                      tileparams);
    }else if(infiles->magfileformat==ALT_LINE_DATA){
      ReadAltLineFile(&mag,&dummy,infiles->magfile,linelen,nlines,
                      tileparams);
    }else if(infiles->magfileformat==ALT_SAMPLE_DATA){
      ReadAltSampFile(&mag,&dummy,infiles->magfile,linelen,nlines,
                      tileparams);
    }
  }
  return(0);
}

/* function: ReadByteMask()
 * ------------------------
 * Read signed byte mask value; set magnitude to zero where byte mask
 * is zero or where pixel is close enough to edge as defined by
 * edgemask parameters; leave magnitude unchanged otherwise.
 */
int ReadByteMask(Array2D<float>& mag, infileT *infiles, long linelen, long nlines,
                 tileparamT *tileparams, paramT *params){

  long row, col, nrow, ncol, fullrow, fullcol;

  /* set up */
  Array2D<signed char> bytemask;
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;

  /* read byte mask (memory allocated by read function) */
  if(strlen(infiles->bytemaskfile)){
    fprintf(sp1,"Reading byte mask from file %s\n",infiles->bytemaskfile);
    Read2DArray(&bytemask,infiles->bytemaskfile,linelen,nlines,
                tileparams,sizeof(signed char *),sizeof(signed char));
  }

  /* loop over rows and columns and zero out magnitude where mask is zero */
  /* also mask edges according to edgemask parameters */
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      fullrow=tileparams->firstrow+row;
      fullcol=tileparams->firstcol+col;
      if((bytemask.size() && bytemask(row,col)==0)
         || fullrow<params->edgemasktop
         || fullcol<params->edgemaskleft
         || fullrow>=nlines-params->edgemaskbot
         || fullcol>=linelen-params->edgemaskright){
        mag(row,col)=0;
      }
    }
  }

  /* done */
  return(0);

}


/* function: ReadUnwrappedEstimateFile()
 * -------------------------------------
 * Reads the unwrapped-phase estimate from a file (assumes file name exists).
 */
int ReadUnwrappedEstimateFile(Array2D<float>* unwrappedestptr, infileT *infiles,
                              long linelen, long nlines,
                              paramT *params, tileparamT *tileparams){

  long nrow, ncol;


  /* initialize */
  Array2D<float> dummy;
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;

  /* read data */
  fprintf(sp1,"Reading coarse unwrapped estimate from file %s\n",
          infiles->estfile);
  if(infiles->estfileformat==ALT_LINE_DATA){
    ReadAltLineFilePhase(unwrappedestptr,infiles->estfile,
                         linelen,nlines,tileparams);
  }else if(infiles->estfileformat==FLOAT_DATA){
    Read2DArray(unwrappedestptr,infiles->estfile,linelen,nlines,
                tileparams,sizeof(float *),sizeof(float));
  }else if(infiles->estfileformat==ALT_SAMPLE_DATA){
    ReadAltSampFile(&dummy,unwrappedestptr,infiles->estfile,
                    linelen,nlines,tileparams);
  }else{
    fflush(NULL);
    fprintf(sp0,"Illegal file format specification for file %s\nAbort\n",
            infiles->estfile);
  }

  /* make sure data is valid */
  if(!ValidDataArray(*unwrappedestptr,nrow,ncol)){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Infinity or NaN found in file " + std::string(infiles->estfile));
  }

  /* flip the sign of the unwrapped estimate if the flip flag is set */
  FlipPhaseArraySign(*unwrappedestptr,params,nrow,ncol);

  /* done */
  return(0);

}


/* function: ReadWeightsFile()
 * ---------------------------
 * Read in weights form rowcol format file of short ints.
 */
int ReadWeightsFile(Array2D<short>* weightsptr,char *weightfile,
                    long linelen, long nlines, tileparamT *tileparams){

  long row, col, nrow, ncol;
  signed char printwarning;


  /* set up and read data */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(strlen(weightfile)){
    fprintf(sp1,"Reading weights from file %s\n",weightfile);
    Read2DRowColFile(weightsptr,weightfile,linelen,nlines,
                     tileparams,sizeof(short));
    auto rowweight=weightsptr->topRows(nrow-1);
    printwarning=FALSE;
    for(row=0;row<nrow-1;row++){
      for(col=0;col<ncol;col++){
        if(rowweight(row,col)<0){
          rowweight(row,col)=0;
          printwarning=TRUE;
        }
      }
    }
    auto colweight=weightsptr->bottomRows(nrow);
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol-1;col++){
        if(colweight(row,col)<0){
          colweight(row,col)=0;
          printwarning=TRUE;
        }
      }
    }
    if(printwarning){
      fflush(NULL);
      fprintf(sp0,"WARNING: Weights must be nonnegative.  Clipping to 0\n");
    }
  }else{
    fprintf(sp1,"No weight file specified.  Assuming uniform weights\n");
    *weightsptr=MakeRowColArray2D<short>(nrow, ncol);
    auto rowweight=weightsptr->topRows(nrow-1);
    auto colweight=weightsptr->bottomRows(nrow);
    Set2DShortArray(rowweight,nrow-1,ncol,DEF_WEIGHT);
    Set2DShortArray(colweight,nrow,ncol-1,DEF_WEIGHT);
  }

  /* done */
  return(0);

}


/* function: ReadIntensity()
 * -------------------------
 * Reads the intensity information from specified file(s).  If possilbe,
 * sets arrays for average power and individual powers of single-pass
 * SAR images.
 */
int ReadIntensity(Array2D<float>* pwrptr, Array2D<float>* pwr1ptr, Array2D<float>* pwr2ptr,
                  infileT *infiles, long linelen, long nlines,
                  paramT *params, tileparamT *tileparams){

  long row, col, nrow, ncol;

  /* initialize */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  Array2D<float> pwr, pwr1, pwr2;

  /* read the data */
  if(strlen(infiles->ampfile2)){

    /* data is given in two separate files */
    fprintf(sp1,"Reading brightness data from files %s and %s\n",
            infiles->ampfile,infiles->ampfile2);
    if(infiles->ampfileformat==FLOAT_DATA){
      Read2DArray(&pwr1,infiles->ampfile,linelen,nlines,tileparams,
                  sizeof(float *),sizeof(float));
      Read2DArray(&pwr2,infiles->ampfile2,linelen,nlines,tileparams,
                  sizeof(float *),sizeof(float));
    }else{
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Illegal file formats specified for files " +
              std::string(infiles->ampfile) + ", " +
              std::string(infiles->ampfile2));
    }

  }else{

    /* data is in single file */
    fprintf(sp1,"Reading brightness data from file %s\n",infiles->ampfile);
    if(infiles->ampfileformat==ALT_SAMPLE_DATA){
      ReadAltSampFile(&pwr1,&pwr2,infiles->ampfile,linelen,nlines,
                      tileparams);
    }else if(infiles->ampfileformat==ALT_LINE_DATA){
      ReadAltLineFile(&pwr1,&pwr2,infiles->ampfile,linelen,nlines,
                      tileparams);
    }else if(infiles->ampfileformat==FLOAT_DATA){
      Read2DArray(&pwr,infiles->ampfile,linelen,nlines,tileparams,
                  sizeof(float *),sizeof(float));
    }else{
      fflush(NULL);
      throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
              "Illegal file format specified for file " +
              std::string(infiles->ampfile));
    }
  }

  /* check data validity */
  if((pwr1.size() && !ValidDataArray(pwr1,nrow,ncol))
     || (pwr2.size() && !ValidDataArray(pwr2,nrow,ncol))
     || (pwr.size() && !ValidDataArray(pwr,nrow,ncol))){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Infinity or NaN found in amplitude or power data");
  }
  if((pwr1.size() && !NonNegDataArray(pwr1,nrow,ncol))
     || (pwr2.size() && !NonNegDataArray(pwr2,nrow,ncol))
     || (pwr.size() && !NonNegDataArray(pwr,nrow,ncol))){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Negative value found in amplitude or power data");
  }

  /* if data is amplitude, square to get power */
  if(params->amplitude){
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        if(pwr1.size() && pwr2.size()){
          pwr1(row,col)*=pwr1(row,col);
          pwr2(row,col)*=pwr2(row,col);
        }else{
          pwr(row,col)*=pwr(row,col);
        }
      }
    }
  }

  /* get the average power */
  if(pwr1.size() && pwr2.size()){
    if(!pwr.size()){
      pwr=Array2D<float>(nrow, ncol);
    }
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        pwr(row,col)=(pwr1(row,col)+pwr2(row,col))/2.0;
      }
    }
  }
  
  /* set output pointers */
  *pwrptr=pwr;
  *pwr1ptr=pwr1;
  *pwr2ptr=pwr2;

  /* done */
  return(0);

}


/* function: ReadCorrelation()
 * ---------------------------
 * Reads the correlation information from specified file.
 */
int ReadCorrelation(Array2D<float>* corrptr, infileT *infiles,
                    long linelen, long nlines, tileparamT *tileparams){

  /* initialize */
  Array2D<float> corr, dummy;

  /* read the data */
  fprintf(sp1,"Reading correlation data from file %s\n",infiles->corrfile);
  if(infiles->corrfileformat==ALT_SAMPLE_DATA){
    ReadAltSampFile(&dummy,&corr,infiles->corrfile,linelen,nlines,tileparams);
  }else if(infiles->corrfileformat==ALT_LINE_DATA){
    ReadAltLineFilePhase(&corr,infiles->corrfile,linelen,nlines,tileparams);
  }else if(infiles->corrfileformat==FLOAT_DATA){
    Read2DArray(&corr,infiles->corrfile,linelen,nlines,tileparams,
                sizeof(float *),sizeof(float));
  }else{
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "Illegal file format specified for file " +
            std::string(infiles->corrfile));
  }

  /* set output pointer */
  *corrptr=corr;

  /* done */
  return(0);

}


/* function: ReadAltLineFile()
 * ---------------------------
 * Read in the data from a file containing magnitude and phase
 * data.  File should have one line of magnitude data, one line
 * of phase data, another line of magnitude data, etc.  
 * ncol refers to the number of complex elements in one line of
 * data.
 */
int ReadAltLineFile(Array2D<float>* mag, Array2D<float>* phase, char *alfile,
                    long linelen, long nlines, tileparamT *tileparams){

  FILE *fp;
  long filesize,row,nrow,ncol,padlen;

  /* open the file */
  if((fp=fopen(alfile,"r"))==NULL){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Can't open file " + std::string(alfile));
  }

  /* get number of lines based on file size and line length */ 
  fseek(fp,0,SEEK_END);            
  filesize=ftell(fp);
  if(filesize!=(2*nlines*linelen*sizeof(float))){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "File " + std::string(alfile) + " wrong size (" +
            std::to_string(nlines) + "x" + std::to_string(linelen) +
            " array expected)");
  }
  fseek(fp,0,SEEK_SET);                 

  /* get memory */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(!mag->size()){
    *mag=Array2D<float>(nrow,ncol);
  }
  if(!phase->size()){
    *phase=Array2D<float>(nrow,ncol);
  }

  /* read the data */
  fseek(fp,(tileparams->firstrow*2*linelen+tileparams->firstcol)
        *sizeof(float),SEEK_CUR);
  padlen=(linelen-ncol)*sizeof(float);
  for(row=0; row<nrow; row++){
    if(fread(mag->row(row).data(),sizeof(float),ncol,fp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while reading from file " + std::string(alfile));
    }
    fseek(fp,padlen,SEEK_CUR);
    if(fread(phase->row(row).data(),sizeof(float),ncol,fp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while reading from file " + std::string(alfile));
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);


  /* done */
  return(0);

}


/* function: ReadAltLineFilePhase()
 * --------------------------------
 * Read only the phase data from a file containing magnitude and phase
 * data.  File should have one line of magnitude data, one line
 * of phase data, another line of magnitude data, etc.
 * ncol refers to the number of complex elements in one line of
 * data.
 */
int ReadAltLineFilePhase(Array2D<float>* phase, char *alfile,
                         long linelen, long nlines, tileparamT *tileparams){

  FILE *fp;
  long filesize,row,nrow,ncol,padlen;

  /* open the file */
  if((fp=fopen(alfile,"r"))==NULL){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Can't open file " + std::string(alfile));
  }

  /* get number of lines based on file size and line length */
  fseek(fp,0,SEEK_END);
  filesize=ftell(fp);
  if(filesize!=(2*nlines*linelen*sizeof(float))){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "File " + std::string(alfile) + " wrong size (" +
            std::to_string(nlines) + "x" + std::to_string(linelen) +
            " array expected)");
  }
  fseek(fp,0,SEEK_SET);

  /* get memory */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(!phase->size()){
    *phase=Array2D<float>(nrow,ncol);
  }

  /* read the phase data */
  fseek(fp,(tileparams->firstrow*2*linelen+linelen
            +tileparams->firstcol)*sizeof(float),SEEK_CUR);
  padlen=(2*linelen-ncol)*sizeof(float);
  for(row=0; row<nrow; row++){
    if(fread(phase->row(row).data(),sizeof(float),ncol,fp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while reading from file " + std::string(alfile));
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);

  /* done */
  return(0);
}


/* function: ReadComplexFile()
 * ---------------------------
 * Reads file of complex floats of the form real,imag,real,imag...
 * ncol is the number of complex samples (half the number of real
 * floats per line).  Ensures that phase values are in the range
 * [0,2pi).
 */
int ReadComplexFile(Array2D<float>* mag, Array2D<float>* phase, char *rifile,
                    long linelen, long nlines, tileparamT *tileparams){

  FILE *fp;
  long filesize,ncol,nrow,row,col,padlen;

  /* open the file */
  if((fp=fopen(rifile,"r"))==NULL){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Can't open file " + std::string(rifile));
  }

  /* get number of lines based on file size and line length */ 
  fseek(fp,0,SEEK_END);
  filesize=ftell(fp);
  if(filesize!=(2*nlines*linelen*sizeof(float))){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "File " + std::string(rifile) + " wrong size (" +
            std::to_string(nlines) + "x" + std::to_string(linelen) +
            " array expected)");
  }
  fseek(fp,0,SEEK_SET);                 

  /* get memory */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(!mag->size()){
    *mag=Array2D<float>(nrow,ncol);
  }
  if(!phase->size()){
    *phase=Array2D<float>(nrow,ncol);
  }
  auto inpline=Array1D<float>(2*ncol);

  /* read the data and convert to magnitude and phase */
  fseek(fp,(tileparams->firstrow*linelen+tileparams->firstcol)
        *2*sizeof(float),SEEK_CUR);
  padlen=(linelen-ncol)*2*sizeof(float);
  for(row=0; row<nrow; row++){
    if(fread(inpline.data(),sizeof(float),2*ncol,fp)!=2*ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while reading from file " + std::string(rifile));
    }
    for(col=0; col<ncol; col++){
      (*mag)(row,col)=sqrt(inpline[2*col]*inpline[2*col]
                            +inpline[2*col+1]*inpline[2*col+1]);
      if(inpline[2*col+1]==0 && inpline[2*col]==0){
        (*phase)(row,col)=0;
      }else if(!IsFinite((*phase)(row,col)=atan2(inpline[2*col+1],
                                                  inpline[2*col]))){
        (*phase)(row,col)=0;
      }else if((*phase)(row,col)<0){
        (*phase)(row,col)+=TWOPI;
      }else if((*phase)(row,col)>=TWOPI){
        (*phase)(row,col)-=TWOPI;
      }
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);

  /* done */
  return(0);
  
}


/* function: ReadAltSampFile()
 * ---------------------------
 * Reads file of real alternating floats from separate images.  Format is
 * real0A, real0B, real1A, real1B, real2A, real2B,...
 * ncol is the number of samples in each image (note the number of
 * floats per line in the specified file).
 */
int ReadAltSampFile(Array2D<float>* arr1, Array2D<float>* arr2, char *infile,
                    long linelen, long nlines, tileparamT *tileparams){

  FILE *fp;
  long filesize,row,col,nrow,ncol,padlen;

  /* open the file */
  if((fp=fopen(infile,"r"))==NULL){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Can't open file " + std::string(infile));
  }

  /* get number of lines based on file size and line length */ 
  fseek(fp,0,SEEK_END);
  filesize=ftell(fp);
  if(filesize!=(2*nlines*linelen*sizeof(float))){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "File " + std::string(infile) + " wrong size (" +
            std::to_string(nlines) + "x" + std::to_string(linelen) +
            " array expected)");
  }
  fseek(fp,0,SEEK_SET);                 

  /* get memory */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(!arr1->size()){
    *arr1=Array2D<float>(nrow,ncol);
  }
  if(!arr2->size()){
    *arr2=Array2D<float>(nrow,ncol);
  }
  auto inpline=Array1D<float>(2*ncol);

  /* read the data */
  fseek(fp,(tileparams->firstrow*linelen+tileparams->firstcol)
        *2*sizeof(float),SEEK_CUR);
  padlen=(linelen-ncol)*2*sizeof(float);
  for(row=0; row<nrow; row++){
    if(fread(inpline.data(),sizeof(float),2*ncol,fp)!=2*ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while reading from file " + std::string(infile));
    }
    for(col=0; col<ncol; col++){
      (*arr1)(row,col)=inpline[2*col];
      (*arr2)(row,col)=inpline[2*col+1];
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);

  /* done */
  return(0);

}


/* function: SetDumpAll()
 * ----------------------
 * Sets names of output files so that the program will dump intermediate
 * arrays.  Only sets names if they are not set already.
 */
int SetDumpAll(outfileT *outfiles, paramT *params){

  if(params->dumpall){
    if(!strlen(outfiles->initfile)){
      StrNCopy(outfiles->initfile,DUMP_INITFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->flowfile)){
      StrNCopy(outfiles->flowfile,DUMP_FLOWFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->eifile)){
      StrNCopy(outfiles->eifile,DUMP_EIFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->rowcostfile)){
      StrNCopy(outfiles->rowcostfile,DUMP_ROWCOSTFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->colcostfile)){
      StrNCopy(outfiles->colcostfile,DUMP_COLCOSTFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->mstrowcostfile)){
      StrNCopy(outfiles->mstrowcostfile,DUMP_MSTROWCOSTFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->mstcolcostfile)){
      StrNCopy(outfiles->mstcolcostfile,DUMP_MSTCOLCOSTFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->mstcostsfile)){
      StrNCopy(outfiles->mstcostsfile,DUMP_MSTCOSTSFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->corrdumpfile)){
      StrNCopy(outfiles->corrdumpfile,DUMP_CORRDUMPFILE,MAXSTRLEN);
    }
    if(!strlen(outfiles->rawcorrdumpfile)){
      StrNCopy(outfiles->rawcorrdumpfile,DUMP_RAWCORRDUMPFILE,MAXSTRLEN);
    }
  }
  return(0);
}


/* function: SetStreamPointers()
 * -----------------------------
 * Sets the default stream pointers (global variables).
 */
int SetStreamPointers(void){

  fflush(NULL);
  if((sp0=DEF_ERRORSTREAM)==NULL){
    if((sp0=fopen(NULLFILE,"w"))==NULL){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Unable to open null file " + std::string(NULLFILE));
    }
  }
  if((sp1=DEF_OUTPUTSTREAM)==NULL){
    if((sp1=fopen(NULLFILE,"w"))==NULL){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Unable to open null file " + std::string(NULLFILE));
    }
  }
  if((sp2=DEF_VERBOSESTREAM)==NULL){
    if((sp2=fopen(NULLFILE,"w"))==NULL){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Unable to open null file " + std::string(NULLFILE));
    }
  }
  if((sp3=DEF_COUNTERSTREAM)==NULL){
    if((sp3=fopen(NULLFILE,"w"))==NULL){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Unable to open null file " + std::string(NULLFILE));
    }
  }
  return(0);
}


/* function: SetVerboseOut()
 * -------------------------
 * Set the global stream pointer sp2 to be stdout if the verbose flag
 * is set in the parameter data type.
 */
int SetVerboseOut(paramT *params){

  fflush(NULL);
  if(params->verbose){
    if(sp2!=stdout && sp2!=stderr && sp2!=stdin && sp2!=NULL){
      fclose(sp2);
    }
    sp2=stdout;
    if(sp3!=stdout && sp3!=stderr && sp3!=stdin && sp3!=NULL){
      fclose(sp3);
    }
    sp3=stdout;
  }
  return(0);
}


/* function: ChildResetStreamPointers()
 * -----------------------------------
 * Reset the global stream pointers for a child.  Streams equal to stdout 
 * are directed to a log file, and errors are written to the screen.
 */
int ChildResetStreamPointers(pid_t pid, long tilerow, long tilecol, 
                             paramT *params){

  FILE *logfp;
  char cwd[MAXSTRLEN]={};

  fflush(NULL);
  const auto logfile=std::string(params->tiledir)+"/"+LOGFILEROOT
    +std::to_string(tilerow)+"_"+std::to_string(tilecol);
  if((logfp=fopen(logfile.c_str(),"w"))==NULL){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Unable to open log file " + logfile);
  }
  fprintf(logfp,"%s (pid %ld): unwrapping tile at row %ld, column %ld\n\n",
          PROGRAMNAME,(long )pid,tilerow,tilecol);
  if(getcwd(cwd,MAXSTRLEN)!=NULL){
    fprintf(logfp,"Current working directory is %s\n",cwd);
  }
  if(sp2==stdout || sp2==stderr){
    sp2=logfp;
  }
  if(sp1==stdout || sp1==stderr){
    sp1=logfp;
  }
  if(sp0==stdout || sp0==stderr){
    sp0=logfp;
  }
  if(sp3!=stdout && sp3!=stderr && sp3!=stdin && sp3!=NULL){
    fclose(sp3);
  }
  if((sp3=fopen(NULLFILE,"w"))==NULL){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Unable to open null file " + std::string(NULLFILE));
  }
  return(0);
}


/* function: DumpIncrCostFiles()
 * -----------------------------
 * Dumps incremental cost arrays, creating file names for them.
 */
int DumpIncrCostFiles(Array2D<incrcostT>& incrcosts, long iincrcostfile,
                      long nflow, long nrow, long ncol){

  long row, col, maxcol;
  char incrcostfile[MAXSTRLEN]={};
  char tempstr[MAXSTRLEN]={};

  /* get memory for tempcosts */
  auto tempcosts=MakeRowColArray2D<short>(nrow, ncol);

  /* create the file names and dump the files */
  /* snprintf() is more elegant, but its unavailable on some machines */
  for(row=0;row<2*nrow-1;row++){
    if(row<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(col=0;col<maxcol;col++){
      tempcosts(row,col)=incrcosts(row,col).poscost;
    }
  }
  strncpy(incrcostfile,INCRCOSTFILEPOS,MAXSTRLEN-1);
  sprintf(tempstr,".%ld_%ld",iincrcostfile,nflow);
  strncat(incrcostfile,tempstr,MAXSTRLEN-strlen(incrcostfile)-1);
  Write2DRowColArray(tempcosts,incrcostfile,
                     nrow,ncol,sizeof(short));
  for(row=0;row<2*nrow-1;row++){
    if(row<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(col=0;col<maxcol;col++){
      tempcosts(row,col)=incrcosts(row,col).negcost;
    }
  }
  strncpy(incrcostfile,INCRCOSTFILENEG,MAXSTRLEN-1);
  sprintf(tempstr,".%ld_%ld",iincrcostfile,nflow);
  strncat(incrcostfile,tempstr,MAXSTRLEN-strlen(incrcostfile)-1);
  Write2DRowColArray(tempcosts,incrcostfile,
                     nrow,ncol,sizeof(short));

  /* done */
  return(0);

}


/* function: MakeTileDir()
 * -----------------------
 * Create a temporary directory for tile files in directory of output file.  
 * Save directory name in buffer in paramT structure.
 */
int MakeTileDir(paramT *params, outfileT *outfiles){

  char path[MAXSTRLEN]={}, basename[MAXSTRLEN]={};
  struct stat statbuf[1]={};
  
  /* create name for tile directory if necessary (use pid to make unique) */
  if(!strlen(params->tiledir)){
    ParseFilename(outfiles->outfile,path,basename);
    const auto tiledir=
      std::string(path)+TMPTILEDIRROOT+std::to_string(params->parentpid);
    std::strcpy(params->tiledir,tiledir.c_str());
  }

  /* return if directory exists */
  /* this is a hack; tiledir could be file or could give other stat() error */
  /*   but if there is a problem, the error will be caught later */
  if(!stat(params->tiledir,statbuf)){
    return(0);
  }

  /* create tile directory */
  fprintf(sp1,"Creating temporary directory %s\n",params->tiledir);
  if(mkdir(params->tiledir,TILEDIRMODE)){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Error creating directory " + std::string(params->tiledir));
  }

  /* done */
  return(0);

}


/* function: SetTileInitOutfile()
 * ------------------------------
 * Set name of temporary tile-mode output assuming nominal output file
 * name is in string passed.  Write new name in string memory pointed
 * to by input.
 */
int SetTileInitOutfile(char *outfile, long pid){

  char path[MAXSTRLEN]={}, basename[MAXSTRLEN]={};
  struct stat statbuf[1]={};
  
  /* create name for output file (use pid to make unique) */
  ParseFilename(outfile,path,basename);
  sprintf(outfile,"%s%s%ld_%s",path,TILEINITFILEROOT,pid,basename);

  /* see if file already exists and exit if so */
  if(!stat(outfile,statbuf)){
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Refusing to write tile init to existing file " +
            std::string(outfile));
  }

  /* done */
  return(0);
  
}


/* function: ParseFilename()
 * -------------------------
 * Given a filename, separates it into path and base filename.  Output
 * buffers should be at least MAXSTRLEN characters, and filename buffer
 * should be no more than MAXSTRLEN characters.  The output path 
 * has a trailing "/" character.
 */
int ParseFilename(const char *filename, char *path, char *basename){

  char tempstring[MAXSTRLEN]={};
  char *tempouttok;

  /* make sure we have a nonzero filename */
  if(!strlen(filename)){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Zero-length filename passed to ParseFilename()");
  }

  /* initialize path */
  if(filename[0]=='/'){
    StrNCopy(path,"/",MAXSTRLEN);
  }else{
    StrNCopy(path,"",MAXSTRLEN);
  }

  /* parse the filename */
  StrNCopy(tempstring,filename,MAXSTRLEN);
  tempouttok=strtok(tempstring,"/");
  while(TRUE){
    StrNCopy(basename,tempouttok,MAXSTRLEN);
    if((tempouttok=strtok(NULL,"/"))==NULL){
      break;
    }
    strcat(path,basename);
    strcat(path,"/");
  }

  /* make sure we have a nonzero base filename */
  if(!strlen(basename)){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Zero-length base filename found in ParseFilename()");
  }

  /* done */
  return(0);

}

} // namespace isce3::unwrap
