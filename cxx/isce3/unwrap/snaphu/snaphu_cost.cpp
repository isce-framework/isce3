/*************************************************************************

  snaphu statistical cost model source file
  Written by Curtis W. Chen
  Copyright 2002 Board of Trustees, Leland Stanford Jr. University
  Please see the supporting documentation for terms of use.
  No warranty.

*************************************************************************/

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <type_traits>

#include <isce3/except/Error.h>

#include "snaphu.h"

namespace isce3::unwrap {

/* static (local) function prototypes */
static
Array2D<costT> BuildStatCostsTopo(Array2D<float>& wrappedphase, Array2D<float>& mag,
                                  Array2D<float>& unwrappedest, Array2D<float>& pwr,
                                  Array2D<float>& corr, Array2D<short>& rowweight, Array2D<short>& colweight,
                                  long nrow, long ncol, tileparamT *tileparams,
                                  outfileT *outfiles, paramT *params);
static
Array2D<costT> BuildStatCostsDefo(Array2D<float>& wrappedphase, Array2D<float>& mag,
                                  Array2D<float>& unwrappedest, Array2D<float>& corr,
                                  Array2D<short>& rowweight, Array2D<short>& colweight,
                                  long nrow, long ncol, tileparamT *tileparams,
                                  outfileT *outfiles, paramT *params);
static
Array2D<smoothcostT> BuildStatCostsSmooth(Array2D<float>& wrappedphase, Array2D<float>& mag,
                                          Array2D<float>& unwrappedest, Array2D<float>& corr,
                                          Array2D<short>& rowweight, Array2D<short>& colweight,
                                          long nrow, long ncol, tileparamT *tileparams,
                                          outfileT *outfiles, paramT *params);
static
void MaskCost(costT *costptr);
static
void MaskSmoothCost(smoothcostT *smoothcostptr);
static
int MaskPrespecifiedArcCosts(Array2D<costT>& costs, Array2D<short>& weights,
                             long nrow, long ncol, paramT *params);
static
int MaskPrespecifiedArcCosts(Array2D<smoothcostT>& costs, Array2D<short>& weights,
                             long nrow, long ncol, paramT *params);
static
int GetIntensityAndCorrelation(Array2D<float>& mag, Array2D<float>& wrappedphase,
                               Array2D<float>* pwrptr, Array2D<float>* corrptr,
                               infileT *infiles, long linelen, long nlines,
                               long nrow, long ncol, outfileT *outfiles,
                               paramT *params, tileparamT *tileparams);
static
int RemoveMean(Array2D<float>& ei, long nrow, long ncol,
               long krowei, long kcolei);
static Array1D<float>
BuildDZRCritLookupTable(double *nominc0ptr, double *dnomincptr,
                        long *tablesizeptr, tileparamT *tileparams,
                        paramT *params);
static
double SolveDZRCrit(double sinnomincangle, double cosnomincangle,
                    paramT *params, double threshold);
static
int SolveEIModelParams(double *slope1ptr, double *slope2ptr, 
                       double *const1ptr, double *const2ptr, 
                       double dzrcrit, double dzr0, double sinnomincangle, 
                       double cosnomincangle, paramT *params);
static
double EIofDZR(double dzr, double sinnomincangle, double cosnomincangle,
               paramT *params);
static Array2D<float>
BuildDZRhoMaxLookupTable(double nominc0, double dnominc,
                         long nominctablesize, double rhomin,
                         double drho, long nrho, paramT *params);
static
double CalcDZRhoMax(double rho, double nominc, paramT *params,
                    double threshold);
static
int CalcInitMaxFlow(paramT *params, Array2D<costT>& costs, long nrow, long ncol);



/* function: BuildCostArrays()
 * ---------------------------
 * Builds cost arrays for arcs based on interferogram intensity
 * and correlation, depending on options and passed parameters.
 */
template<class CostTag>
int BuildCostArrays(Array2D<typename CostTag::Cost>* costsptr, Array2D<short>* mstcostsptr,
                    Array2D<float>& mag, Array2D<float>& wrappedphase,
                    Array2D<float>& unwrappedest, long linelen, long nlines,
                    long nrow, long ncol, paramT *params,
                    tileparamT *tileparams, infileT *infiles,
                    outfileT *outfiles, CostTag tag){

  long row, col, maxcol, tempcost;
  long poscost, negcost, costtypesize;
  Array2D<float> pwr, corr;

  using Cost=typename CostTag::Cost;
  Array2D<Cost> costs;
  Array2D<short> scalarcosts;
  Array2D<bidircostT> bidircosts;

  auto info=pyre::journal::info_t("isce3.unwrap.snaphu");

  /* initializations to silence compiler warnings */
  costtypesize=0;

  /* read weights */
  Array2D<short> weights;
  ReadWeightsFile(&weights,infiles->weightfile,linelen,nlines,tileparams);
  Array2D<short> rowweight=weights.topRows(nrow-1);
  Array2D<short> colweight=weights.bottomRows(nrow);

  /* set weights to zero for arcs adjacent to zero-magnitude pixels */
  if(mag.size()){
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        if(mag(row,col)==0){
          if(row>0){
            rowweight(row-1,col)=0;
          }
          if(row<nrow-1){
            rowweight(row,col)=0;
          }
          if(col>0){
            colweight(row,col-1)=0;
          }
          if(col<ncol-1){
            colweight(row,col)=0;
          }
        }
      }
    }
  }

  /* if we're only initializing and we don't want statistical weights */
  if(params->initonly && params->costmode==NOSTATCOSTS){
    *mstcostsptr=weights;
    return(0);
  }

  /* size of the data type for holding cost data depends on cost mode */
  if(params->costmode==TOPO){
    costtypesize=sizeof(costT);
  }else if(params->costmode==DEFO){
    costtypesize=sizeof(costT);
  }else if(params->costmode==SMOOTH){
    costtypesize=sizeof(smoothcostT);
  }

  /* build or read the statistical cost arrays unless we were told not to */
  if(strlen(infiles->costinfile)){

    /* read cost info from file */
    info << pyre::journal::at(__HERE__)
         << "Reading cost information from file " << infiles->costinfile
         << pyre::journal::endl;
    Read2DRowColFile(&costs,infiles->costinfile,
                     linelen,nlines,tileparams,costtypesize);
    (*costsptr)=costs;

    /* weights of arcs next to masked pixels are set to zero */
    /* make sure corresponding costs are nulled when costs are read from */
    /*   file rather than internally generated since read costs are not */
    /*   multiplied by weights */
    if constexpr(std::is_same<Cost,costT>{} || std::is_same<Cost,smoothcostT>{}){
      MaskPrespecifiedArcCosts(costs,weights,nrow,ncol,params);
    }else{
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Bad Cost type in BuildCostArrays()");
    }

  }else if(params->costmode!=NOSTATCOSTS){

    /* get intensity and correlation info */
    /* correlation generated from interferogram and amplitude if not given */
    GetIntensityAndCorrelation(mag,wrappedphase,&pwr,&corr,infiles,
                               linelen,nlines,nrow,ncol,outfiles,
                               params,tileparams);

    /* call specific functions for building cost array and */
    /* set global pointers to functions for calculating and evaluating costs */
    if(params->costmode==TOPO){
      info << pyre::journal::at(__HERE__)
           << "Calculating topography-mode cost parameters"
           << pyre::journal::endl;
      if constexpr(std::is_same<Cost,costT>{}){
        costs=BuildStatCostsTopo(wrappedphase,mag,unwrappedest,pwr,corr,
                                 rowweight,colweight,nrow,ncol,tileparams,
                                 outfiles,params);
      }else{
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Bad Cost type in BuildCostArrays()");
      }
    }else if(params->costmode==DEFO){
      info << pyre::journal::at(__HERE__)
           << "Calculating deformation-mode cost parameters"
           << pyre::journal::endl;
      if constexpr(std::is_same<Cost,costT>{}){
        costs=BuildStatCostsDefo(wrappedphase,mag,unwrappedest,corr,
                                 rowweight,colweight,nrow,ncol,tileparams,
                                 outfiles,params);
      }else{
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Bad Cost type in BuildCostArrays()");
      }
    }else if(params->costmode==SMOOTH){
      info << pyre::journal::at(__HERE__)
           << "Calculating smooth-solution cost parameters"
           << pyre::journal::endl;
      if constexpr(std::is_same<Cost,smoothcostT>{}){
        costs=BuildStatCostsSmooth(wrappedphase,mag,unwrappedest,corr,
                                   rowweight,colweight,nrow,ncol,tileparams,
                                   outfiles,params);
      }else{
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Bad Cost type in BuildCostArrays()");
      }
    }else{
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(), "unrecognized cost mode");
    }
    (*costsptr)=costs;
    

  }/* end if(params->costmode!=NOSTATCOSTS) */
  
  Array2D<Cost> rowcost=costs.block(0,0,nrow-1,ncol);
  Array2D<Cost> colcost=costs.block(nrow-1,0,nrow,ncol-1);

  /* dump statistical cost arrays */
  if(strlen(infiles->costinfile) || params->costmode!=NOSTATCOSTS){
    if(strlen(outfiles->costoutfile)){
      Write2DRowColArray(costs,outfiles->costoutfile,
                        nrow,ncol,costtypesize);
    }else{
      if(strlen(outfiles->rowcostfile)){
        Write2DArray(rowcost,outfiles->rowcostfile,
                     nrow-1,ncol,costtypesize);
      }
      if(strlen(outfiles->colcostfile)){
        Write2DArray(colcost,outfiles->colcostfile,
                     nrow,ncol-1,costtypesize);
      }
    }
  }

  /* get memory for scalar costs if in Lp mode */
  if(params->p>=0){
    if(params->bidirlpn){
      if constexpr(std::is_same<Cost,bidircostT>{}){
        bidircosts=MakeRowColArray2D<bidircostT>(nrow,ncol);
        (*costsptr)=bidircosts;
      }else{
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Bad Cost type in BuildCostArrays()");
      }
    }else{
      if constexpr(std::is_same<Cost,short>{}){
        scalarcosts=MakeRowColArray2D<short>(nrow,ncol);
        (*costsptr)=scalarcosts;
      }else{
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Bad Cost type in BuildCostArrays()");
      }
    }
  }

  /* now, set scalar costs for MST initialization or optimization if needed */
  if(params->costmode==NOSTATCOSTS){    

    /* if in no-statistical-costs mode, copy weights to scalarcosts array */
    if(!params->initonly){
      for(row=0;row<2*nrow-1;row++){
        if(row<nrow-1){
          maxcol=ncol;
        }else{
          maxcol=ncol-1;
        }
        for(col=0;col<maxcol;col++){
          if(params->bidirlpn){
            bidircosts(row,col).posweight=weights(row,col);
            bidircosts(row,col).negweight=weights(row,col);
          }else{
            scalarcosts(row,col)=weights(row,col);
          }
        }
      }
    }

    /* unless input is already unwrapped, use weights memory for mstcosts */
    if(!params->unwrapped){
      *mstcostsptr = weights;
    }else{
      *mstcostsptr = Array2D<short>{};
    }

  }else if(!params->unwrapped || params->p>=0){

    /* if we got here, we had statistical costs and we need scalar weights */
    /*   from them for MST initialization or for Lp optimization */
    for(row=0;row<2*nrow-1;row++){
      if(row<nrow-1){
        maxcol=ncol;
      }else{
        maxcol=ncol-1;
      }
      for(col=0;col<maxcol;col++){

        /* calculate incremental costs for flow=0, nflow=1 */
        CalcCost(costs,0,row,col,1,nrow,params,
                 &poscost,&negcost,tag);

        /* take smaller of positive and negative incremental cost */
        if(poscost<negcost){
          tempcost=poscost;
        }else{
          tempcost=negcost;
        }

        /* clip scalar cost so it is between 1 and params->maxcost */
        /* note: weights used for MST algorithm will not be zero along */
        /*   masked edges since they are clipped to 1, but MST is run */
        /*   once on entire network, not just non-masked regions */
        weights(row,col)=LClip(tempcost,MINSCALARCOST,params->maxcost);

        /* assign Lp costs if in Lp mode */
        /* let scalar cost be zero if costs in both directions are zero */
        if(params->p>=0){
          if(params->bidirlpn){
            bidircosts(row,col).posweight=LClip(poscost,0,params->maxcost);
            bidircosts(row,col).negweight=LClip(negcost,0,params->maxcost);
          }else{
            scalarcosts(row,col)=weights(row,col);
            if(poscost==0 && negcost==0){
              scalarcosts(row,col)=0;
            }
          }
        }
      }
    }

    /* set costs for corner arcs to prevent ambiguous flows */
    weights(nrow-1,0)=LARGESHORT;
    weights(nrow-1,ncol-2)=LARGESHORT;
    weights(2*nrow-2,0)=LARGESHORT;
    weights(2*nrow-2,ncol-2)=LARGESHORT;
    if(params->p>=0){
      if(params->bidirlpn){
        bidircosts(nrow-1,0).posweight=LARGESHORT;
        bidircosts(nrow-1,0).negweight=LARGESHORT;
        bidircosts(nrow-1,ncol-2).posweight=LARGESHORT;
        bidircosts(nrow-1,ncol-2).negweight=LARGESHORT;
        bidircosts(2*nrow-2,0).posweight=LARGESHORT;
        bidircosts(2*nrow-2,0).negweight=LARGESHORT;
        bidircosts(2*nrow-2,ncol-2).posweight=LARGESHORT;
        bidircosts(2*nrow-2,ncol-2).negweight=LARGESHORT;
      }else{
        scalarcosts(nrow-1,0)=LARGESHORT;
        scalarcosts(nrow-1,ncol-2)=LARGESHORT;
        scalarcosts(2*nrow-2,0)=LARGESHORT;
        scalarcosts(2*nrow-2,ncol-2)=LARGESHORT;
      }
    }

    /* dump mst initialization costs */
    if(strlen(outfiles->mstrowcostfile)){
      Write2DArray(rowweight,outfiles->mstrowcostfile,
                   nrow-1,ncol,sizeof(short));
    }
    if(strlen(outfiles->mstcolcostfile)){
      Write2DArray(colweight,outfiles->mstcolcostfile,
                   nrow,ncol-1,sizeof(short));
    }
    if(strlen(outfiles->mstcostsfile)){
      Write2DRowColArray(rowweight,outfiles->mstcostsfile,
                         nrow,ncol,sizeof(short));
    }

    /* unless input is unwrapped, calculate initialization max flow */
    if(params->initmaxflow==AUTOCALCSTATMAX && !params->unwrapped){
      if constexpr(std::is_same<Cost,costT>{}){
        CalcInitMaxFlow(params,costs,nrow,ncol);
      }else{
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Bad Cost type in BuildCostArrays()");
      }
    }

    /* use memory allocated for weights arrays for mstcosts if needed */
    if(!params->unwrapped){
      *mstcostsptr = weights;
    }
  }

  /* done */
  return(0);
  
}


/* function: BuildStatCostsTopo()
 * ------------------------------
 * Builds statistical cost arrays for topography mode.
 */
static
Array2D<costT> BuildStatCostsTopo(Array2D<float>& wrappedphase, Array2D<float>& /*mag*/,
                                  Array2D<float>& unwrappedest, Array2D<float>& pwr,
                                  Array2D<float>& corr, Array2D<short>& rowweight, Array2D<short>& colweight,
                                  long nrow, long ncol, tileparamT *tileparams,
                                  outfileT *outfiles, paramT *params){

  long row, col, iei, nrho, nominctablesize;
  long kperpdpsi, kpardpsi, sigsqshortmin;
  double a, re, dr, slantrange, nearrange, nominc0, dnominc;
  double nomincangle, nomincind, sinnomincangle, cosnomincangle, bperp;
  double baseline, baselineangle, lambda, lookangle;
  double dzlay, dzei, dzr0, dzrcrit, dzeimin, dphilaypeak, dzrhomax;
  double azdzfactor, dzeifactor, dzeiweight, dzlayfactor;
  double avgei, eicrit, layminei, laywidth, slope1, const1, slope2, const2;
  double rho, rho0, rhomin, drho, rhopow;
  double sigsqrho, sigsqrhoconst, sigsqei, sigsqlay;
  double glay, costscale, ambiguityheight, ztoshort, ztoshortsq;
  double nshortcycle, midrangeambight;
  signed char noshadow, nolayover;

  Array2D<float> ei;

  constexpr int output_detail_level=2;
  auto verbose=pyre::journal::info_t("isce3.unwrap.snaphu",output_detail_level);

  /* get memory and set cost array pointers */
  auto costs=MakeRowColArray2D<costT>(nrow,ncol);
  auto rowcost=costs.block(0,0,nrow-1,ncol);
  auto colcost=costs.block(nrow-1,0,nrow,ncol-1);

  /* set up */
  rho0=(params->rhosconst1)/(params->ncorrlooks)+(params->rhosconst2);
  rhomin=params->rhominfactor*rho0;
  rhopow=2*(params->cstd1)+(params->cstd2)*log(params->ncorrlooks)
    +(params->cstd3)*(params->ncorrlooks);
  sigsqshortmin=params->sigsqshortmin;
  kperpdpsi=params->kperpdpsi;
  kpardpsi=params->kpardpsi;
  dr=params->dr;
  nearrange=params->nearrange+dr*tileparams->firstcol;
  drho=params->drho;
  nrho=(long )floor((1-rhomin)/drho)+1;
  nshortcycle=params->nshortcycle;
  layminei=params->layminei;
  laywidth=params->laywidth;
  azdzfactor=params->azdzfactor;
  dzeifactor=params->dzeifactor;
  dzeiweight=params->dzeiweight;
  dzeimin=params->dzeimin;
  dzlayfactor=params->dzlayfactor;
  sigsqei=params->sigsqei;
  lambda=params->lambda;
  noshadow=!(params->shadow);
  a=params->orbitradius;
  re=params->earthradius;

  /* despeckle the interferogram intensity */
  verbose << pyre::journal::at(__HERE__)
          << "Despeckling intensity image"
          << pyre::journal::endl;
  Despeckle(pwr,&ei,nrow,ncol);

  /* remove large-area average intensity */
  verbose << pyre::journal::at(__HERE__)
          << "Normalizing intensity"
          << pyre::journal::endl;
  RemoveMean(ei,nrow,ncol,params->krowei,params->kcolei);

  /* dump normalized, despeckled intensity */
  if(strlen(outfiles->eifile)){
    Write2DArray(ei,outfiles->eifile,nrow,ncol,sizeof(float));
  }

  /* compute some midswath parameters */
  slantrange=nearrange+ncol/2*dr;
  sinnomincangle=sin(acos((a*a-slantrange*slantrange-re*re)
                          /(2*slantrange*re)));
  lookangle=asin(re/a*sinnomincangle);

  /* see if we were passed bperp rather than baseline and baselineangle */
  if(params->bperp){
    if(params->bperp>0){
      params->baselineangle=lookangle;
    }else{
      params->baselineangle=lookangle+PI;
    }
    params->baseline=fabs(params->bperp);
  }

  /* the baseline should be halved if we are in single antenna transmit mode */
  if(params->transmitmode==SINGLEANTTRANSMIT){
    params->baseline/=2.0;
  }
  baseline=params->baseline;
  baselineangle=params->baselineangle;

  /* build lookup table for dzrcrit vs incidence angle */
  auto dzrcrittable=BuildDZRCritLookupTable(&nominc0,&dnominc,&nominctablesize,
                                            tileparams,params);

  /* build lookup table for dzrhomax vs incidence angle */
  auto dzrhomaxtable=BuildDZRhoMaxLookupTable(nominc0,dnominc,nominctablesize,
                                              rhomin,drho,nrho,params);

  /* set cost autoscale factor based on midswath parameters */
  bperp=baseline*cos(lookangle-baselineangle);
  midrangeambight=fabs(lambda*slantrange*sinnomincangle/(2*bperp));
  costscale=params->costscale*fabs(params->costscaleambight/midrangeambight);
  glay=-costscale*log(params->layconst);

  /* get memory for wrapped difference arrays */
  auto dpsi=Array2D<float>(nrow,ncol);
  auto avgdpsi=Array2D<float>(nrow,ncol);

  /* build array of mean wrapped phase differences in range */
  /* simple average of phase differences is biased, but mean phase */
  /*   differences usually near zero, so don't bother with complex average */
  verbose << pyre::journal::at(__HERE__)
          << "Building range cost arrays"
          << pyre::journal::endl;
  CalcWrappedRangeDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
                        nrow,ncol);

  /* build colcost array (range slopes) */
  /* loop over range */
  for(col=0;col<ncol-1;col++){

    /* compute range dependent parameters */
    slantrange=nearrange+col*dr;
    cosnomincangle=(a*a-slantrange*slantrange-re*re)/(2*slantrange*re);
    nomincangle=acos(cosnomincangle);
    sinnomincangle=sin(nomincangle);
    lookangle=asin(re/a*sinnomincangle);
    dzr0=-dr*cosnomincangle;
    bperp=baseline*cos(lookangle-baselineangle);
    ambiguityheight=-(lambda*slantrange*sinnomincangle)/(2*bperp);
    sigsqrhoconst=2.0*ambiguityheight*ambiguityheight/12.0;  
    ztoshort=nshortcycle/ambiguityheight;
    ztoshortsq=ztoshort*ztoshort;
    sigsqlay=ambiguityheight*ambiguityheight*params->sigsqlayfactor;

    /* interpolate scattering model parameters */
    nomincind=(nomincangle-nominc0)/dnominc;
    dzrcrit=LinInterp1D(dzrcrittable,nomincind,nominctablesize);
    SolveEIModelParams(&slope1,&slope2,&const1,&const2,dzrcrit,dzr0,
                       sinnomincangle,cosnomincangle,params);
    eicrit=(dzrcrit-const1)/slope1;
    dphilaypeak=params->dzlaypeak/ambiguityheight;

    /* loop over azimuth */
    for(row=0;row<nrow;row++){

      /* see if we have a masked pixel */
      if(colweight(row,col)==0){

        /* masked pixel */
        MaskCost(&colcost(row,col));

      }else{

        /* topography-mode costs */

        /* calculate variance due to decorrelation */
        /* factor of 2 in sigsqrhoconst for pdf convolution */
        rho=corr(row,col);
        if(rho<rhomin){
          rho=0;
        }
        sigsqrho=sigsqrhoconst*pow(1-rho,rhopow);

        /* calculate dz expected from EI if no layover */
        if(ei(row,col)>eicrit){
          dzei=(slope2*ei(row,col)+const2)*dzeifactor;
        }else{
          dzei=(slope1*ei(row,col)+const1)*dzeifactor;
        }
        if(noshadow && dzei<dzeimin){
          dzei=dzeimin;
        }

        /* calculate dz expected from EI if layover exists */
        dzlay=0;
        if(ei(row,col)>layminei){
          for(iei=0;iei<laywidth;iei++){
            if(ei(row,col+iei)>eicrit){
              dzlay+=slope2*ei(row,col+iei)+const2;
            }else{
              dzlay+=slope1*ei(row,col+iei)+const1;
            }
            if(col+iei>ncol-2){
              break;
            }
          }
        }
        if(dzlay){
          dzlay=(dzlay+iei*(-2.0*dzr0))*dzlayfactor;
        }
          
        /* set maximum dz based on unbiased correlation and layover max */ 
        if(rho>0){
          dzrhomax=LinInterp2D(dzrhomaxtable,nomincind,(rho-rhomin)/drho,
                               nominctablesize,nrho);
          if(dzrhomax<dzlay){  
            dzlay=dzrhomax;
          }
        }

        /* set cost parameters in terms of flow, represented as shorts */
        nolayover=TRUE;
        if(dzlay){
          if(rho>0){
            colcost(row,col).offset=nshortcycle*
              (dpsi(row,col)-0.5*(avgdpsi(row,col)+dphilaypeak));
          }else{
            colcost(row,col).offset=nshortcycle*
              (dpsi(row,col)-0.25*avgdpsi(row,col)-0.75*dphilaypeak);
          }
          colcost(row,col).sigsq=(sigsqrho+sigsqei+sigsqlay)*ztoshortsq
            /(costscale*colweight(row,col));
          if(colcost(row,col).sigsq<sigsqshortmin){
            colcost(row,col).sigsq=sigsqshortmin;
          }
          colcost(row,col).dzmax=dzlay*ztoshort;
          colcost(row,col).laycost=colweight(row,col)*glay;
          if(labs(colcost(row,col).dzmax)
             >floor(sqrt(colcost(row,col).laycost*colcost(row,col).sigsq))){
            nolayover=FALSE;
          }
        }
        if(nolayover){
          colcost(row,col).sigsq=(sigsqrho+sigsqei)*ztoshortsq
            /(costscale*colweight(row,col));
          if(colcost(row,col).sigsq<sigsqshortmin){
            colcost(row,col).sigsq=sigsqshortmin;
          }
          if(rho>0){
            colcost(row,col).offset=ztoshort*
              (ambiguityheight*(dpsi(row,col)-0.5*avgdpsi(row,col))
               -0.5*dzeiweight*dzei);
          }else{
            colcost(row,col).offset=ztoshort*
              (ambiguityheight*(dpsi(row,col)-0.25*avgdpsi(row,col))
               -0.75*dzeiweight*dzei);
          }
          colcost(row,col).laycost=NOCOSTSHELF;
          colcost(row,col).dzmax=LARGESHORT;
        }

        /* shift PDF to account for flattening by coarse unwrapped estimate */
        if(unwrappedest.size()){
          colcost(row,col).offset+=(nshortcycle/TWOPI*
                                     (unwrappedest(row,col+1)
                                      -unwrappedest(row,col)));
        }

      }
    }
  } /* end of range gradient cost calculation */

  /* reset layover constant for row (azimuth) costs */
  glay+=(-costscale*log(azdzfactor)); 

  /* build array of mean wrapped phase differences in azimuth */
  /* biased, but not much, so don't bother with complex averaging */
  verbose << pyre::journal::at(__HERE__)
          << "Building azimuth cost arrays"
          << pyre::journal::endl;
  CalcWrappedAzDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
                     nrow,ncol);
  
  /* build rowcost array */
  /* for the rowcost array, there is symmetry between positive and */
  /*   negative flows, so we average ei[][] and corr[][] values in azimuth */
  /* loop over range */
  for(col=0;col<ncol;col++){

    /* compute range dependent parameters */
    slantrange=nearrange+col*dr;
    cosnomincangle=(a*a-slantrange*slantrange-re*re)/(2*slantrange*re);
    nomincangle=acos(cosnomincangle);
    sinnomincangle=sin(nomincangle);
    lookangle=asin(re/a*sinnomincangle);
    dzr0=-dr*cosnomincangle;
    bperp=baseline*cos(lookangle-baselineangle);
    ambiguityheight=-lambda*slantrange*sinnomincangle/(2*bperp);
    sigsqrhoconst=2.0*ambiguityheight*ambiguityheight/12.0;  
    ztoshort=nshortcycle/ambiguityheight;
    ztoshortsq=ztoshort*ztoshort;
    sigsqlay=ambiguityheight*ambiguityheight*params->sigsqlayfactor;

    /* interpolate scattering model parameters */
    nomincind=(nomincangle-nominc0)/dnominc;
    dzrcrit=LinInterp1D(dzrcrittable,nomincind,nominctablesize);
    SolveEIModelParams(&slope1,&slope2,&const1,&const2,dzrcrit,dzr0,
                       sinnomincangle,cosnomincangle,params);
    eicrit=(dzrcrit-const1)/slope1;
    dphilaypeak=params->dzlaypeak/ambiguityheight;

    /* loop over azimuth */
    for(row=0;row<nrow-1;row++){

      /* see if we have a masked pixel */
      if(rowweight(row,col)==0){

        /* masked pixel */
        MaskCost(&rowcost(row,col));

      }else{

        /* topography-mode costs */

        /* variance due to decorrelation */
        /* get correlation and clip small values because of estimator bias */
        rho=(corr(row,col)+corr(row+1,col))/2.0;
        if(rho<rhomin){
          rho=0;
        }
        sigsqrho=sigsqrhoconst*pow(1-rho,rhopow);

        /* if no layover, the expected dz for azimuth will always be 0 */
        dzei=0;

        /* calculate dz expected from EI if layover exists */
        dzlay=0;
        avgei=(ei(row,col)+ei(row+1,col))/2.0;
        if(avgei>layminei){
          for(iei=0;iei<laywidth;iei++){
            avgei=(ei(row,col+iei)+ei(row+1,col+iei))/2.0;
            if(avgei>eicrit){
              dzlay+=slope2*avgei+const2;
            }else{
              dzlay+=slope1*avgei+const1;
            }
            if(col+iei>ncol-2){
              break;
            }
          }
        }
        if(dzlay){
          dzlay=(dzlay+iei*(-2.0*dzr0))*dzlayfactor;
        }
          
        /* set maximum dz based on correlation max and layover max */ 
        if(rho>0){
          dzrhomax=LinInterp2D(dzrhomaxtable,nomincind,(rho-rhomin)/drho,
                               nominctablesize,nrho);
          if(dzrhomax<dzlay){
            dzlay=dzrhomax;
          }
        }

        /* set cost parameters in terms of flow, represented as shorts */
        if(rho>0){
          rowcost(row,col).offset=nshortcycle*
            (dpsi(row,col)-avgdpsi(row,col));
        }else{
          rowcost(row,col).offset=nshortcycle*
            (dpsi(row,col)-0.5*avgdpsi(row,col));
        }
        nolayover=TRUE;
        if(dzlay){
          rowcost(row,col).sigsq=(sigsqrho+sigsqei+sigsqlay)*ztoshortsq
            /(costscale*rowweight(row,col));
          if(rowcost(row,col).sigsq<sigsqshortmin){
            rowcost(row,col).sigsq=sigsqshortmin;
          }
          rowcost(row,col).dzmax=fabs(dzlay*ztoshort);
          rowcost(row,col).laycost=rowweight(row,col)*glay;
          if(labs(rowcost(row,col).dzmax)
             >floor(sqrt(rowcost(row,col).laycost*rowcost(row,col).sigsq))){
            nolayover=FALSE;
          }
        }
        if(nolayover){
          rowcost(row,col).sigsq=(sigsqrho+sigsqei)*ztoshortsq
            /(costscale*rowweight(row,col));
          if(rowcost(row,col).sigsq<sigsqshortmin){
            rowcost(row,col).sigsq=sigsqshortmin;
          }
          rowcost(row,col).laycost=NOCOSTSHELF;
          rowcost(row,col).dzmax=LARGESHORT;
        }

        /* shift PDF to account for flattening by coarse unwrapped estimate */
        if(unwrappedest.size()){
          rowcost(row,col).offset+=(nshortcycle/TWOPI*
                                     (unwrappedest(row+1,col)
                                      -unwrappedest(row,col)));
        }

      }
    }
  }  /* end of azimuth gradient cost calculation */

  /* return pointer to costs arrays */
  return(costs);

}


/* function: BuildStatCostsDefo()
 * ------------------------------
 * Builds statistical cost arrays for deformation mode.
 */
static
Array2D<costT> BuildStatCostsDefo(Array2D<float>& wrappedphase, Array2D<float>& /*mag*/,
                                  Array2D<float>& unwrappedest, Array2D<float>& corr,
                                  Array2D<short>& rowweight, Array2D<short>& colweight,
                                  long nrow, long ncol, tileparamT * /*tileparams*/,
                                  outfileT * /*outfiles*/, paramT *params){

  long row, col;
  long kperpdpsi, kpardpsi, sigsqshortmin, defomax;
  double rho, rho0, rhopow;
  double defocorrthresh, sigsqcorr, sigsqrho, sigsqrhoconst;
  double glay, costscale;
  double nshortcycle, nshortcyclesq;

  constexpr int output_detail_level=2;
  auto verbose=pyre::journal::info_t("isce3.unwrap.snaphu",output_detail_level);

  /* get memory and set cost array pointers */
  auto costs=MakeRowColArray2D<costT>(nrow,ncol);
  auto rowcost=costs.block(0,0,nrow-1,ncol);
  auto colcost=costs.block(nrow-1,0,nrow,ncol-1);

  /* set up */
  rho0=(params->rhosconst1)/(params->ncorrlooks)+(params->rhosconst2);
  defocorrthresh=params->defothreshfactor*rho0;
  rhopow=2*(params->cstd1)+(params->cstd2)*log(params->ncorrlooks)
    +(params->cstd3)*(params->ncorrlooks);
  sigsqrhoconst=2.0/12.0;
  sigsqcorr=params->sigsqcorr;
  sigsqshortmin=params->sigsqshortmin;
  kperpdpsi=params->kperpdpsi;
  kpardpsi=params->kpardpsi;
  costscale=params->costscale; 
  nshortcycle=params->nshortcycle;
  nshortcyclesq=nshortcycle*nshortcycle;
  glay=-costscale*log(params->defolayconst);
  defomax=(long )ceil(params->defomax*nshortcycle);

  /* get memory for wrapped difference arrays */
  auto dpsi=Array2D<float>(nrow,ncol);
  auto avgdpsi=Array2D<float>(nrow,ncol);

  /* build array of mean wrapped phase differences in range */
  /* simple average of phase differences is biased, but mean phase */
  /*   differences usually near zero, so don't bother with complex average */
  verbose << pyre::journal::at(__HERE__)
          << "Building range cost arrays"
          << pyre::journal::endl;
  CalcWrappedRangeDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
                        nrow,ncol);

  /* build colcost array (range slopes) */
  for(col=0;col<ncol-1;col++){
    for(row=0;row<nrow;row++){

      /* see if we have a masked pixel */
      if(colweight(row,col)==0){

        /* masked pixel */
        MaskCost(&colcost(row,col));

      }else{

        /* deformation-mode costs */

        /* calculate variance due to decorrelation */
        /* need symmetry for range if deformation */
        rho=(corr(row,col)+corr(row,col+1))/2.0;
        if(rho<defocorrthresh){
          rho=0;
        }
        sigsqrho=(sigsqrhoconst*pow(1-rho,rhopow)+sigsqcorr)*nshortcyclesq;

        /* set cost paramaters in terms of flow, represented as shorts */
        if(rho>0){
          colcost(row,col).offset=nshortcycle*
            (dpsi(row,col)-avgdpsi(row,col));
        }else{
          colcost(row,col).offset=nshortcycle*
            (dpsi(row,col)-0.5*avgdpsi(row,col));
        }
        colcost(row,col).sigsq=sigsqrho/(costscale*colweight(row,col));
        if(colcost(row,col).sigsq<sigsqshortmin){
          colcost(row,col).sigsq=sigsqshortmin;
        }
        if(rho<defocorrthresh){
          colcost(row,col).dzmax=defomax;
          colcost(row,col).laycost=colweight(row,col)*glay;
          if(colcost(row,col).dzmax<floor(sqrt(colcost(row,col).laycost
                                                *colcost(row,col).sigsq))){
            colcost(row,col).laycost=NOCOSTSHELF;
            colcost(row,col).dzmax=LARGESHORT;
          }
        }else{
          colcost(row,col).laycost=NOCOSTSHELF;
          colcost(row,col).dzmax=LARGESHORT;
        }
      }

      /* shift PDF to account for flattening by coarse unwrapped estimate */
      if(unwrappedest.size()){
        colcost(row,col).offset+=(nshortcycle/TWOPI*
                                   (unwrappedest(row,col+1)
                                    -unwrappedest(row,col)));
      }
    }
  }  /* end of range gradient cost calculation */

  /* build array of mean wrapped phase differences in azimuth */
  /* biased, but not much, so don't bother with complex averaging */
  verbose << pyre::journal::at(__HERE__)
          << "Building azimuth cost arrays"
          << pyre::journal::endl;
  CalcWrappedAzDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
                     nrow,ncol);

  /* build rowcost array */
  for(col=0;col<ncol;col++){
    for(row=0;row<nrow-1;row++){

      /* see if we have a masked pixel */
      if(rowweight(row,col)==0){

        /* masked pixel */
        MaskCost(&rowcost(row,col));

      }else{

        /* deformation-mode costs */

        /* variance due to decorrelation */
        /* get correlation and clip small values because of estimator bias */
        rho=(corr(row,col)+corr(row+1,col))/2.0;
        if(rho<defocorrthresh){
          rho=0;
        }
        sigsqrho=(sigsqrhoconst*pow(1-rho,rhopow)+sigsqcorr)*nshortcyclesq;

        /* set cost paramaters in terms of flow, represented as shorts */
        if(rho>0){
          rowcost(row,col).offset=nshortcycle*
            (dpsi(row,col)-avgdpsi(row,col));
        }else{
          rowcost(row,col).offset=nshortcycle*
            (dpsi(row,col)-0.5*avgdpsi(row,col));
        }
        rowcost(row,col).sigsq=sigsqrho/(costscale*rowweight(row,col));
        if(rowcost(row,col).sigsq<sigsqshortmin){
          rowcost(row,col).sigsq=sigsqshortmin;
        }
        if(rho<defocorrthresh){
          rowcost(row,col).dzmax=defomax;
          rowcost(row,col).laycost=rowweight(row,col)*glay;
          if(rowcost(row,col).dzmax<floor(sqrt(rowcost(row,col).laycost
                                                *rowcost(row,col).sigsq))){
            rowcost(row,col).laycost=NOCOSTSHELF;
            rowcost(row,col).dzmax=LARGESHORT;
          }
        }else{
          rowcost(row,col).laycost=NOCOSTSHELF;
          rowcost(row,col).dzmax=LARGESHORT;
        }
      }

      /* shift PDF to account for flattening by coarse unwrapped estimate */
      if(unwrappedest.size()){
        rowcost(row,col).offset+=(nshortcycle/TWOPI*
                                   (unwrappedest(row+1,col)
                                    -unwrappedest(row,col)));
      }
    }
  } /* end of azimuth cost calculation */

  /* return pointer to costs arrays */
  return(costs);

}


/* function: BuildStatCostsSmooth()
 * --------------------------------
 * Builds statistical cost arrays for smooth-solution mode.
 */
static
Array2D<smoothcostT> BuildStatCostsSmooth(Array2D<float>& wrappedphase, Array2D<float>& /*mag*/,
                                          Array2D<float>& unwrappedest, Array2D<float>& corr,
                                          Array2D<short>& rowweight, Array2D<short>& colweight,
                                          long nrow, long ncol, tileparamT * /*tileparams*/,
                                          outfileT * /*outfiles*/, paramT *params){

  long row, col;
  long kperpdpsi, kpardpsi, sigsqshortmin;
  double rho, rho0, rhopow;
  double defocorrthresh, sigsqcorr, sigsqrho, sigsqrhoconst;
  double costscale;
  double nshortcycle, nshortcyclesq;

  constexpr int output_detail_level=2;
  auto verbose=pyre::journal::info_t("isce3.unwrap.snaphu",output_detail_level);

  /* get memory and set cost array pointers */
  auto costs=MakeRowColArray2D<smoothcostT>(nrow,ncol);
  auto rowcost=costs.block(0,0,nrow-1,ncol);
  auto colcost=costs.block(nrow-1,0,nrow,ncol-1);

  /* set up */
  rho0=(params->rhosconst1)/(params->ncorrlooks)+(params->rhosconst2);
  defocorrthresh=params->defothreshfactor*rho0;
  rhopow=2*(params->cstd1)+(params->cstd2)*log(params->ncorrlooks)
    +(params->cstd3)*(params->ncorrlooks);
  sigsqrhoconst=2.0/12.0;
  sigsqcorr=params->sigsqcorr;
  sigsqshortmin=params->sigsqshortmin;
  kperpdpsi=params->kperpdpsi;
  kpardpsi=params->kpardpsi;
  costscale=params->costscale; 
  nshortcycle=params->nshortcycle;
  nshortcyclesq=nshortcycle*nshortcycle;

  /* get memory for wrapped difference arrays */
  auto dpsi=Array2D<float>(nrow,ncol);
  auto avgdpsi=Array2D<float>(nrow,ncol);

  /* build array of mean wrapped phase differences in range */
  /* simple average of phase differences is biased, but mean phase */
  /*   differences usually near zero, so don't bother with complex average */
  verbose << pyre::journal::at(__HERE__)
          << "Building range cost arrays"
          << pyre::journal::endl;
  CalcWrappedRangeDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
                        nrow,ncol);

  /* build colcost array (range slopes) */
  for(col=0;col<ncol-1;col++){
    for(row=0;row<nrow;row++){

      /* see if we have a masked pixel */
      if(colweight(row,col)==0){

        /* masked pixel */
        MaskSmoothCost(&colcost(row,col));

      }else{

        /* smooth-mode costs */

        /* calculate variance due to decorrelation */
        /* need symmetry for range if deformation */
        rho=(corr(row,col)+corr(row,col+1))/2.0;
        if(rho<defocorrthresh){
          rho=0;
        }
        sigsqrho=(sigsqrhoconst*pow(1-rho,rhopow)+sigsqcorr)*nshortcyclesq;

        /* set cost paramaters in terms of flow, represented as shorts */
        if(rho>0){
          colcost(row,col).offset=nshortcycle*
            (dpsi(row,col)-avgdpsi(row,col));
        }else{
          colcost(row,col).offset=nshortcycle*
            (dpsi(row,col)-0.5*avgdpsi(row,col));
        }
        colcost(row,col).sigsq=sigsqrho/(costscale*colweight(row,col));
        if(colcost(row,col).sigsq<sigsqshortmin){
          colcost(row,col).sigsq=sigsqshortmin;
        }
      }

      /* shift PDF to account for flattening by coarse unwrapped estimate */
      if(unwrappedest.size()){
        colcost(row,col).offset+=(nshortcycle/TWOPI*
                                   (unwrappedest(row,col+1)
                                    -unwrappedest(row,col)));
      }
    }
  }  /* end of range gradient cost calculation */

  /* build array of mean wrapped phase differences in azimuth */
  /* biased, but not much, so don't bother with complex averaging */
  verbose << pyre::journal::at(__HERE__)
          << "Building azimuth cost arrays"
          << pyre::journal::endl;
  CalcWrappedAzDiffs(dpsi,avgdpsi,wrappedphase,kperpdpsi,kpardpsi,
                     nrow,ncol);

  /* build rowcost array */
  for(col=0;col<ncol;col++){
    for(row=0;row<nrow-1;row++){

      /* see if we have a masked pixel */
      if(rowweight(row,col)==0){

        /* masked pixel */
        MaskSmoothCost(&rowcost(row,col));

      }else{

        /* smooth-mode costs */

        /* variance due to decorrelation */
        /* get correlation and clip small values because of estimator bias */
        rho=(corr(row,col)+corr(row+1,col))/2.0;
        if(rho<defocorrthresh){
          rho=0;
        }
        sigsqrho=(sigsqrhoconst*pow(1-rho,rhopow)+sigsqcorr)*nshortcyclesq;

        /* set cost paramaters in terms of flow, represented as shorts */
        if(rho>0){
          rowcost(row,col).offset=nshortcycle*
            (dpsi(row,col)-avgdpsi(row,col));
        }else{
          rowcost(row,col).offset=nshortcycle*
            (dpsi(row,col)-0.5*avgdpsi(row,col));
        }
        rowcost(row,col).sigsq=sigsqrho/(costscale*rowweight(row,col));
        if(rowcost(row,col).sigsq<sigsqshortmin){
          rowcost(row,col).sigsq=sigsqshortmin;
        }
      }

      /* shift PDF to account for flattening by coarse unwrapped estimate */
      if(unwrappedest.size()){
        rowcost(row,col).offset+=(nshortcycle/TWOPI*
                                   (unwrappedest(row+1,col)
                                    -unwrappedest(row,col)));
      }
    }
  } /* end of azimuth cost calculation */

  /* return pointer to costs arrays */
  return(costs);

}


/* function: MaskCost()
 * --------------------
 * Set values of costT structure pointed to by input pointer to give zero
 * cost, as for arcs next to masked pixels.
 */
static
void MaskCost(costT *costptr){

  /* set to special values */
  costptr->laycost=0;
  costptr->offset=LARGESHORT/2;
  costptr->dzmax=LARGESHORT;
  costptr->sigsq=LARGESHORT;

}


/* function: MaskSmoothCost()
 * --------------------------
 * Set values of smoothcostT structure pointed to by input pointer to give zero
 * cost, as for arcs next to masked pixels.
 */
static
void MaskSmoothCost(smoothcostT *smoothcostptr){

  /* set to special values */
  smoothcostptr->offset=LARGESHORT/2;
  smoothcostptr->sigsq=LARGESHORT;

}


/* function: MaskPrespecifiedArcCosts()
 * ------------------------------------
 * Loop over grid arcs and set costs to null if corresponding weights
 * are null.
 */
static
int MaskPrespecifiedArcCosts(Array2D<costT>& costs, Array2D<short>& weights,
                             long nrow, long ncol, paramT */*params*/){

  long row, col, maxcol;

  /* loop over all arcs */
  for(row=0;row<2*nrow-1;row++){
    if(row<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(col=0;col<maxcol;col++){
      if(weights(row,col)==0){
          MaskCost(&costs(row,col));
      }
    }
  }

  /* done */
  return(0);

}


/* function: MaskPrespecifiedArcCosts()
 * ------------------------------------
 * Loop over grid arcs and set costs to null if corresponding weights
 * are null.
 */
static
int MaskPrespecifiedArcCosts(Array2D<smoothcostT>& costs, Array2D<short>& weights,
                             long nrow, long ncol, paramT */*params*/){

  long row, col, maxcol;

  /* loop over all arcs */
  for(row=0;row<2*nrow-1;row++){
    if(row<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(col=0;col<maxcol;col++){
      if(weights(row,col)==0){
          MaskSmoothCost(&costs(row,col));
      }
    }
  }

  /* done */
  return(0);

}


/* function: GetIntensityAndCorrelation()
 * --------------------------------------
 * Reads amplitude and correlation info from files if specified.  If ampfile
 * not given, uses interferogram magnitude.  If correlation file not given,
 * generates correlatin info from interferogram and amplitude.
 */
static
int GetIntensityAndCorrelation(Array2D<float>& mag, Array2D<float>& wrappedphase,
                               Array2D<float>* pwrptr, Array2D<float>* corrptr,
                               infileT *infiles, long linelen, long nlines,
                               long nrow, long ncol, outfileT *outfiles,
                               paramT *params, tileparamT *tileparams){

  long row, col, krowcorr, kcolcorr, iclipped;
  double rho0, rhomin, biaseddefaultcorr;

  Array2D<float> pwr, pwr1, pwr2, corr;

  auto info=pyre::journal::info_t("isce3.unwrap.snaphu");

  /* read intensity, if specified */
  if(strlen(infiles->ampfile)){
    ReadIntensity(&pwr,&pwr1,&pwr2,infiles,linelen,nlines,params,tileparams);
  }else{
    if(params->costmode==TOPO){
      info << pyre::journal::at(__HERE__)
           << "No brightness file specified. "
           << "Using interferogram magnitude as intensity"
           << pyre::journal::endl;
    }
    pwr = Array2D<float>(nrow, ncol);
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        pwr(row,col)=mag(row,col);
      }
    }
  }

  /* read corrfile, if specified */
  if(strlen(infiles->corrfile)){
    ReadCorrelation(&corr,infiles,linelen,nlines,tileparams);
  }else if(pwr1.size() && pwr2.size() && params->havemagnitude){

    /* generate the correlation info from the interferogram and amplitude */
    info << pyre::journal::at(__HERE__)
         << "Generating correlation from interferogram and intensity"
         << pyre::journal::endl;

    /* get the correct number of looks, and make sure its odd */
    krowcorr=1+2*floor(params->ncorrlooksaz/(double )params->nlooksaz/2);
    kcolcorr=1+2*floor(params->ncorrlooksrange/(double )params->nlooksrange/2);

    /* calculate equivalent number of independent looks */
    params->ncorrlooks=(kcolcorr*(params->dr/params->rangeres))
      *(krowcorr*(params->da/params->azres))*params->nlooksother;
    info << pyre::journal::at(__HERE__)
         << "   (" << std::fixed << std::setprecision(1)
         << params->ncorrlooks << " equivalent independent looks)"
         << pyre::journal::endl;

    /* get real and imaginary parts of interferogram */
    auto realcomp = Array2D<float>(nrow, ncol);
    auto imagcomp = Array2D<float>(nrow, ncol);
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        realcomp(row,col)=mag(row,col)*cos(wrappedphase(row,col));
        imagcomp(row,col)=mag(row,col)*sin(wrappedphase(row,col));
      }
    }

    /* do complex spatial averaging on the interferogram */
    Array2D<float> padreal, padimag;
    try {
      padreal=MirrorPad(realcomp,nrow,ncol,(krowcorr-1)/2,(kcolcorr-1)/2);
      padimag=MirrorPad(imagcomp,nrow,ncol,(krowcorr-1)/2,(kcolcorr-1)/2);
    } catch (const isce3::except::RuntimeError&) {
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Correlation averaging box too large for input array size");
    }
    auto avgreal = realcomp;
    BoxCarAvg(avgreal,padreal,nrow,ncol,krowcorr,kcolcorr);
    auto avgimag = imagcomp;
    BoxCarAvg(avgimag,padimag,nrow,ncol,krowcorr,kcolcorr);

    /* spatially average individual SAR power images */
    auto padpwr1=MirrorPad(pwr1,nrow,ncol,(krowcorr-1)/2,(kcolcorr-1)/2);
    auto avgpwr1 = pwr1;
    BoxCarAvg(avgpwr1,padpwr1,nrow,ncol,krowcorr,kcolcorr);
    auto padpwr2=MirrorPad(pwr2,nrow,ncol,(krowcorr-1)/2,(kcolcorr-1)/2);
    auto avgpwr2 = pwr2;
    BoxCarAvg(avgpwr2,padpwr2,nrow,ncol,krowcorr,kcolcorr);

    /* build correlation data */
    corr = Array2D<float>(nrow, ncol);
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        if(avgpwr1(row,col)<=0 || avgpwr2(row,col)<=0){
          corr(row,col)=0.0;
        }else{
          corr(row,col)=sqrt((avgreal(row,col)*avgreal(row,col)
                               +avgimag(row,col)*avgimag(row,col))
                              /(avgpwr1(row,col)*avgpwr2(row,col)));
        }
      }
    }

  }else{

    /* no file specified: set corr to default value */
    /* find biased default correlation using */
    /* inverse of unbias method used by BuildCostArrays() */
    corr = Array2D<float>(nrow, ncol);
    info << pyre::journal::at(__HERE__)
         << "No correlation file specified. Assuming correlation = "
         << params->defaultcorr << pyre::journal::endl;
    rho0=(params->rhosconst1)/(params->ncorrlooks)+(params->rhosconst2);
    rhomin=params->rhominfactor*rho0;
    if(params->defaultcorr>rhomin){
      biaseddefaultcorr=params->defaultcorr;
    }else{
      biaseddefaultcorr=0.0;
    }
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        corr(row,col)=biaseddefaultcorr;
      }
    }
  }

  /* dump correlation data if necessary */
  if(strlen(outfiles->rawcorrdumpfile)){
    Write2DArray(corr,outfiles->rawcorrdumpfile,
                 nrow,ncol,sizeof(float));
  }

  /* check correlation data validity */
  iclipped=0;
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      if(!IsFinite(corr(row,col))){
        fflush(NULL);
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "NaN or infinity found in correlation data");
      }else if(corr(row,col)>1.0){
        if(corr(row,col)>1.001){
          iclipped++;               /* don't warn for minor numerical errors */
        }
        corr(row,col)=1.0;
      }else if(corr(row,col)<0.0){
        if(corr(row,col)<-0.001){
          iclipped++;               /* don't warn for minor numerical errors */
        }
        corr(row,col)=0.0;
      }
    }
  }
  if(iclipped){
    fflush(NULL);
    auto warnings=pyre::journal::warning_t("isce3.unwrap.snaphu");
    warnings << pyre::journal::at(__HERE__)
             << "WARNING: " << iclipped
             << " illegal correlation values clipped to [0,1]"
             << pyre::journal::endl;
  }

  /* dump correlation data if necessary */
  if(strlen(outfiles->corrdumpfile)){
    Write2DArray(corr,outfiles->corrdumpfile,
                 nrow,ncol,sizeof(float));
  }

  /* set output pointers */
  *pwrptr=pwr;
  *corrptr=corr;

  /* done */
  return(0);
  
}


/* function: RemoveMean()
 * -------------------------
 * Divides intensity by average over sliding window.
 */
static
int RemoveMean(Array2D<float>& ei, long nrow, long ncol,
               long krowei, long kcolei){

  long row, col;

  /* make sure krowei, kcolei are odd */
  if(!(krowei % 2)){
    krowei++;
  }
  if(!(kcolei % 2)){
    kcolei++;
  }

  /* get memory */
  auto avgei = Array2D<float>(nrow, ncol);

  /* pad ei in new array */
  Array2D<float> padei;
  try {
    padei=MirrorPad(ei,nrow,ncol,(krowei-1)/2,(kcolei-1)/2);
  } catch (const isce3::except::RuntimeError&) {
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Intensity-normalization averaging box too large for input array "
            "size");
  }

  /* calculate average ei by using sliding window */
  BoxCarAvg(avgei,padei,nrow,ncol,krowei,kcolei);

  /* divide ei by avgei */
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      ei(row,col)/=(avgei(row,col));
    }
  }

  return(0);

}


/* function: BuildDZRCritLookupTable()
 * -----------------------------------
 * Builds a 1-D lookup table of dzrcrit values indexed by incidence angle
 * (in rad).
 */
static Array1D<float>
BuildDZRCritLookupTable(double *nominc0ptr, double *dnomincptr,
                        long *tablesizeptr, tileparamT *tileparams,
                        paramT *params){

  long tablesize, k;
  double nominc, nominc0, nomincmax, dnominc;
  double a, re, slantrange;

  /* compute nominal spherical earth incidence angle for near and far range */
  a=params->orbitradius;
  re=params->earthradius;
  slantrange=params->nearrange+params->dr*tileparams->firstcol;
  nominc0=acos((a*a-slantrange*slantrange-re*re)/(2*slantrange*re));
  slantrange+=params->dr*tileparams->ncol;
  nomincmax=acos((a*a-slantrange*slantrange-re*re)/(2*slantrange*re));
  if(!IsFinite(nominc0) || !IsFinite(nomincmax)){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Geometry error detected. Check altitude, near range, and earth "
            "radius parameters");
  }

  /* build lookup table */
  dnominc=params->dnomincangle;
  tablesize=(long )floor((nomincmax-nominc0)/dnominc)+1;
  auto dzrcrittable = Array1D<float>(tablesize);
  nominc=nominc0;
  for(k=0;k<tablesize;k++){
    dzrcrittable[k]=(float )SolveDZRCrit(sin(nominc),cos(nominc),params,
                                 params->threshold);
    nominc+=dnominc;
    if(nominc>PI/2.0){
      nominc-=dnominc;
    }
  }
  
  /* set return variables */
  (*nominc0ptr)=nominc;
  (*dnomincptr)=dnominc;
  (*tablesizeptr)=tablesize;
  return(dzrcrittable);

}


/* function: SolveDZRCrit()
 * ------------------------
 * Numerically solve for the transition point of the linearized scattering 
 * model.
 */
static
double SolveDZRCrit(double sinnomincangle, double cosnomincangle, 
                    paramT *params, double threshold){

  double residual, thetai, kds, n, dr, dzr, dx;
  double costhetai, cos2thetai, step;
  double dzrcritfactor, diffuse, specular;
  long i;

  /* get parameters */
  kds=params->kds;
  n=params->specularexp;
  dr=params->dr;  
  dzrcritfactor=params->dzrcritfactor;

  /* solve for critical incidence angle */
  thetai=PI/4;
  step=PI/4-1e-6;
  i=0;
  while(TRUE){
    if((cos2thetai=cos(2*thetai))<0){
      cos2thetai=0;
    }
    diffuse=dzrcritfactor*kds*cos(thetai);
    specular=pow(cos2thetai,n);
    if(fabs(residual=diffuse-specular)<threshold*diffuse){
      break;
    }
    if(residual<0){
      thetai+=step;
    }else{
      thetai-=step;
    }
    step/=2.0;
    if(++i>MAXITERATION){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Couldn't find critical incidence angle (check scattering "
              "parameters)");
    }
  }

  /* solve for critical height change */
  costhetai=cos(thetai);
  dzr=params->initdzr;
  step=dzr+dr*cosnomincangle-1e-2;
  i=0;
  while(TRUE){
    dx=(dr+dzr*cosnomincangle)/sinnomincangle;
    if(fabs(residual=costhetai-(dzr*sinnomincangle+dx*cosnomincangle)
            /sqrt(dzr*dzr+dx*dx))
       <threshold*costhetai){
      return(dzr);
    }
    if(residual<0){
      dzr-=step;
    }else{
      dzr+=step;
    }
    step/=2.0;
    if(++i>MAXITERATION){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Couldn't find critical slope (check geometry parameters)");
    }
  }
}


/* function: SolveEIModelParams()
 * ------------------------------
 * Calculates parameters for linearized model of EI vs. range slope
 * relationship.
 */
int SolveEIModelParams(double *slope1ptr, double *slope2ptr, 
                       double *const1ptr, double *const2ptr, 
                       double dzrcrit, double dzr0, double sinnomincangle, 
                       double cosnomincangle, paramT *params){
  
  double slope1, slope2, const1, const2, sloperatio;
  double dzr3, ei3;
  
  /* set up */
  sloperatio=params->kds*params->sloperatiofactor;

  /* find normalized intensity at 15(dzrcrit-dzr0)+dzr0 */
  dzr3=15.0*(dzrcrit-dzr0)+dzr0;
  ei3=EIofDZR(dzr3,sinnomincangle,cosnomincangle,params)
    /EIofDZR(0,sinnomincangle,cosnomincangle,params);

  /* calculate parameters */
  const1=dzr0;
  slope2=(sloperatio*(dzrcrit-const1)-dzrcrit+dzr3)/ei3;
  slope1=slope2/sloperatio;
  const2=dzr3-slope2*ei3;

  /* set return values */
  *slope1ptr=slope1;
  *slope2ptr=slope2;
  *const1ptr=const1;
  *const2ptr=const2;
  return(0);
  
}


/* function: EIofDZR()
 * -------------------
 * Calculates expected value of intensity with arbitrary units for given
 * parameters.  Assumes azimuth slope is zero.
 */
static
double EIofDZR(double dzr, double sinnomincangle, double cosnomincangle,
               paramT *params){

  double dr, da, dx, kds, n, dzr0, projarea;
  double costhetai, cos2thetai, sigma0;

  dr=params->dr;
  da=params->da;
  dx=dr/sinnomincangle+dzr*cosnomincangle/sinnomincangle;
  kds=params->kds;
  n=params->specularexp;
  dzr0=-dr*cosnomincangle;
  projarea=da*fabs((dzr-dzr0)/sinnomincangle);
  costhetai=projarea/sqrt(dzr*dzr*da*da + da*da*dx*dx);
  if(costhetai>SQRTHALF){
    cos2thetai=2*costhetai*costhetai-1;
    sigma0=kds*costhetai+pow(cos2thetai,n);
  }else{
    sigma0=kds*costhetai;
  }
  return(sigma0*projarea);

}


/* function: BuildDZRhoMaxLookupTable()
 * ------------------------------------
 * Builds a 2-D lookup table of dzrhomax values vs nominal incidence angle
 * (rad) and correlation.
 */
static Array2D<float>
BuildDZRhoMaxLookupTable(double nominc0, double dnominc,
                         long nominctablesize, double rhomin,
                         double drho, long nrho, paramT *params){

  long krho, knominc;
  double nominc, rho;

  auto dzrhomaxtable = Array2D<float>(nominctablesize, nrho);
  nominc=nominc0;
  for(knominc=0;knominc<nominctablesize;knominc++){
    rho=rhomin;
    for(krho=0;krho<nrho;krho++){
      dzrhomaxtable(knominc,krho)=(float )CalcDZRhoMax(rho,nominc,params,
                                                       params->threshold);
      rho+=drho;
    }
    nominc+=dnominc;
  }
  return(dzrhomaxtable);

}


/* function: CalcDZRhoMax()
 * ------------------------
 * Calculates the maximum slope (in range) for the given unbiased correlation
 * using spatial decorrelation as an upper limit (Zebker & Villasenor,
 * 1992).
 */
static
double CalcDZRhoMax(double rho, double nominc, paramT *params, 
                    double threshold){

  long i;
  double dx, dr, dz, dzstep, rhos, sintheta, costheta, numerator;
  double a, re, bperp, slantrange, lookangle;
  double costhetairsq, rhosfactor, residual;


  /* set up */
  i=0;
  dr=params->dr;
  costheta=cos(nominc);
  sintheta=sin(nominc);
  dzstep=params->initdzstep;
  a=params->orbitradius;
  re=params->earthradius;
  lookangle=asin(re/a*sintheta);
  bperp=params->baseline*cos(lookangle-params->baselineangle);
  slantrange=sqrt(a*a+re*re-2*a*re*cos(nominc-lookangle));
  rhosfactor=2.0*fabs(bperp)*(params->rangeres)/((params->lambda)*slantrange);

  /* take care of the extremes */
  if(rho>=1.0){
    return(-dr*costheta);
  }else if(rho<=0){
    return(LARGEFLOAT);
  }

  /* start with slope for unity correlation, step slope upwards */
  dz=-dr*costheta;
  rhos=1.0;
  while(rhos>rho){
    dz+=dzstep;
    dx=(dr+dz*costheta)/sintheta;
    numerator=dz*sintheta+dx*costheta;
    costhetairsq=numerator*numerator/(dz*dz+dx*dx);
    rhos=1-rhosfactor*sqrt(costhetairsq/(1-costhetairsq));
    if(rhos<0){
      rhos=0;
    }
    if(dz>BIGGESTDZRHOMAX){
      return(BIGGESTDZRHOMAX);
    }
  }

  /* now iteratively decrease step size and narrow in on correct slope */
  while(fabs(residual=rhos-rho)>threshold*rho){
    dzstep/=2.0;
    if(residual<0){
      dz-=dzstep;
    }else{
      dz+=dzstep;
    }
    dx=(dr+dz*costheta)/sintheta;
    numerator=dz*sintheta+dx*costheta;
    costhetairsq=numerator*numerator/(dz*dz+dx*dx);
    rhos=1-rhosfactor*sqrt(costhetairsq/(1-costhetairsq));
    if(rhos<0){
      rhos=0;
    }
    if(++i>MAXITERATION){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Couldn't find slope for correlation of " + std::to_string(rho) +
              "(check geometry and spatial decorrelation parameters)");
    }
  }

  return(dz);
}


/* function: CalcCost()
 * ------------------------
 * Calculates topography arc distance given an array of cost data structures.
 */
void CalcCost(Array2D<costT>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, TopoCostTag /*tag*/){

  long idz1, idz2pos, idz2neg, cost1, nflowsq, poscost, negcost;
  long nshortcycle, layfalloffconst;
  long offset, sigsq, laycost, dzmax;
  costT *cost;

  /* get arc info */
  cost=&costs(arcrow,arccol);
  offset=cost->offset;
  sigsq=cost->sigsq;
  dzmax=cost->dzmax;
  laycost=cost->laycost;

  /* just return 0 if we have zero cost arc */
  if(sigsq==LARGESHORT){
    (*poscostptr)=0;
    (*negcostptr)=0;
    return;
  }

  /* compute argument to cost function */
  nshortcycle=params->nshortcycle;
  layfalloffconst=params->layfalloffconst;
  if(arcrow<nrow-1){

    /* row cost: dz symmetric with respect to origin */
    idz1=labs(flow*nshortcycle+offset);
    idz2pos=labs((flow+nflow)*nshortcycle+offset);
    idz2neg=labs((flow-nflow)*nshortcycle+offset);

  }else{

    /* column cost: non-symmetric dz */
    /* dzmax will only be < 0 if we have a column arc */
    if(dzmax<0){
      dzmax*=-1;
      idz1=-(flow*nshortcycle+offset);
      idz2pos=-((flow+nflow)*nshortcycle+offset);
      idz2neg=-((flow-nflow)*nshortcycle+offset);
    }else{
      idz1=flow*nshortcycle+offset;
      idz2pos=(flow+nflow)*nshortcycle+offset;
      idz2neg=(flow-nflow)*nshortcycle+offset;
    }

  }

  /* calculate cost1 */
  if(idz1>dzmax){
    idz1-=dzmax;
    cost1=(idz1*idz1)/(layfalloffconst*sigsq)+laycost; 
  }else{
    cost1=(idz1*idz1)/sigsq;
    if(laycost!=NOCOSTSHELF && idz1>0 && cost1>laycost){
      cost1=laycost;
    }
  }

  /* calculate positive cost increment */
  if(idz2pos>dzmax){
    idz2pos-=dzmax;
    poscost=(idz2pos*idz2pos)/(layfalloffconst*sigsq)
      +laycost-cost1;
  }else{
    poscost=(idz2pos*idz2pos)/sigsq;
    if(laycost!=NOCOSTSHELF && idz2pos>0 && poscost>laycost){
      poscost=laycost-cost1;
    }else{
      poscost-=cost1;
    }
  }

  /* calculate negative cost increment */
  if(idz2neg>dzmax){
    idz2neg-=dzmax;
    negcost=(idz2neg*idz2neg)/(layfalloffconst*sigsq)
      +laycost-cost1;
  }else{
    negcost=(idz2neg*idz2neg)/sigsq;
    if(laycost!=NOCOSTSHELF && idz2neg>0 && negcost>laycost){
      negcost=laycost-cost1;
    }else{
      negcost-=cost1;
    }
  }

  /* scale costs for this nflow */
  nflowsq=nflow*nflow;
  if(poscost>0){
    *poscostptr=(long )ceil((double )poscost/nflowsq);
  }else{
    *poscostptr=(long )floor((double )poscost/nflowsq);
  }
  if(negcost>0){
    *negcostptr=(long )ceil((double )negcost/nflowsq);
  }else{
    *negcostptr=(long )floor((double )negcost/nflowsq);
  }

  /* done */
  return;
  
}


/* function: CalcCost()
 * ------------------------
 * Calculates deformation arc distance given an array of cost data structures.
 */
void CalcCost(Array2D<costT>& costs, long flow, long arcrow, long arccol,
              long nflow, long /*nrow*/, paramT *params,
              long *poscostptr, long *negcostptr, DefoCostTag /*tag*/){

  long idz1, idz2pos, idz2neg, cost1, nflowsq, poscost, negcost;
  long nshortcycle, layfalloffconst;
  costT *cost;

  /* get arc info */
  cost=&costs(arcrow,arccol);

  /* just return 0 if we have zero cost arc */
  if(cost->sigsq==LARGESHORT){
    (*poscostptr)=0;
    (*negcostptr)=0;
    return;
  }

  /* compute argument to cost function */
  nshortcycle=params->nshortcycle;
  layfalloffconst=params->layfalloffconst;
  idz1=labs(flow*nshortcycle+cost->offset);
  idz2pos=labs((flow+nflow)*nshortcycle+cost->offset);
  idz2neg=labs((flow-nflow)*nshortcycle+cost->offset);

  /* calculate cost1 */
  if(idz1>cost->dzmax){
    idz1-=cost->dzmax;
    cost1=(idz1*idz1)/(layfalloffconst*(cost->sigsq))+cost->laycost; 
  }else{
    cost1=(idz1*idz1)/cost->sigsq;
    if(cost->laycost!=NOCOSTSHELF && cost1>cost->laycost){
      cost1=cost->laycost;
    }
  }

  /* calculate positive cost increment */
  if(idz2pos>cost->dzmax){
    idz2pos-=cost->dzmax;
    poscost=(idz2pos*idz2pos)/(layfalloffconst*(cost->sigsq))
      +cost->laycost-cost1;
  }else{
    poscost=(idz2pos*idz2pos)/cost->sigsq;
    if(cost->laycost!=NOCOSTSHELF && poscost>cost->laycost){
      poscost=cost->laycost-cost1;
    }else{
      poscost-=cost1;
    }
  }

  /* calculate negative cost increment */
  if(idz2neg>cost->dzmax){
    idz2neg-=cost->dzmax;
    negcost=(idz2neg*idz2neg)/(layfalloffconst*(cost->sigsq))
      +cost->laycost-cost1;
  }else{
    negcost=(idz2neg*idz2neg)/cost->sigsq;
    if(cost->laycost!=NOCOSTSHELF && negcost>cost->laycost){
      negcost=cost->laycost-cost1;
    }else{
      negcost-=cost1;
    }
  }

  /* scale costs for this nflow */
  nflowsq=nflow*nflow;
  if(poscost>0){
    *poscostptr=(long )ceil((double )poscost/nflowsq);
  }else{
    *poscostptr=(long )floor((double )poscost/nflowsq);
  }
  if(negcost>0){
    *negcostptr=(long )ceil((double )negcost/nflowsq);
  }else{
    *negcostptr=(long )floor((double )negcost/nflowsq);
  }

  /* done */
  return;

}


/* function: CalcCost()
 * --------------------------
 * Calculates smooth-solution arc distance given an array of smoothcost
 *  data structures.
 */
void CalcCost(Array2D<smoothcostT>& costs, long flow, long arcrow, long arccol,
              long nflow, long /*nrow*/, paramT *params,
              long *poscostptr, long *negcostptr, SmoothCostTag /*tag*/){

  long idz1, idz2pos, idz2neg, cost1, nflowsq, poscost, negcost;
  long nshortcycle;
  smoothcostT *cost;


  /* get arc info */
  cost=&costs(arcrow,arccol);

  /* just return 0 if we have zero cost arc */
  if(cost->sigsq==LARGESHORT){
    (*poscostptr)=0;
    (*negcostptr)=0;
    return;
  }

  /* compute argument to cost function */
  nshortcycle=params->nshortcycle;
  idz1=labs(flow*nshortcycle+cost->offset);
  idz2pos=labs((flow+nflow)*nshortcycle+cost->offset);
  idz2neg=labs((flow-nflow)*nshortcycle+cost->offset);

  /* calculate cost1 */
  cost1=(idz1*idz1)/cost->sigsq;

  /* calculate positive cost increment */
  poscost=(idz2pos*idz2pos)/cost->sigsq-cost1;

  /* calculate negative cost increment */
  negcost=(idz2neg*idz2neg)/cost->sigsq-cost1;

  /* scale costs for this nflow */
  nflowsq=nflow*nflow;
  if(poscost>0){
    *poscostptr=(long )ceil((double )poscost/nflowsq);
  }else{
    *poscostptr=(long )floor((double )poscost/nflowsq);
  }
  if(negcost>0){
    *negcostptr=(long )ceil((double )negcost/nflowsq);
  }else{
    *negcostptr=(long )floor((double )negcost/nflowsq);
  }

  /* done */
  return;

}


/* function: CalcCost()
 * ----------------------
 * Calculates the L0 arc distance given an array of short integer weights.
 */
void CalcCost(Array2D<short>& costs, long flow, long arcrow, long arccol,
              long nflow, long /*nrow*/, paramT */*params*/,
              long *poscostptr, long *negcostptr, L0CostTag /*tag*/){

  /* L0-norm */
  if(flow){
    if(flow+nflow){
      *poscostptr=0;
    }else{
      *poscostptr=-costs(arcrow,arccol);
    }
    if(flow-nflow){
      *negcostptr=0;
    }else{
      *negcostptr=-costs(arcrow,arccol);
    }
  }else{
    *poscostptr=costs(arcrow,arccol);
    *negcostptr=costs(arcrow,arccol);
  }

  /* done */
  return;

}


/* function: CalcCost()
 * ----------------------
 * Calculates the L1 arc distance given an array of short integer weights.
 */
void CalcCost(Array2D<short>& costs, long flow, long arcrow, long arccol,
              long nflow, long /*nrow*/, paramT */*params*/,
              long *poscostptr, long *negcostptr, L1CostTag /*tag*/){

  /* L1-norm */
  *poscostptr=costs(arcrow,arccol)*(labs(flow+nflow)-labs(flow));
  *negcostptr=costs(arcrow,arccol)*(labs(flow-nflow)-labs(flow));

  /* done */
  return;

}


/* function: CalcCost()
 * ----------------------
 * Calculates the L2 arc distance given an array of short integer weights.
 */
void CalcCost(Array2D<short>& costs, long flow, long arcrow, long arccol,
              long nflow, long /*nrow*/, paramT */*params*/,
              long *poscostptr, long *negcostptr, L2CostTag /*tag*/){

  long flow2, flowsq;

  /* L2-norm */
  flowsq=flow*flow;
  flow2=flow+nflow;
  *poscostptr=costs(arcrow,arccol)*(flow2*flow2-flowsq);
  flow2=flow-nflow;
  *negcostptr=costs(arcrow,arccol)*(flow2*flow2-flowsq);

  /* done */
  return;
}


/* function: CalcCost()
 * ----------------------
 * Calculates the Lp arc distance given an array of short integer weights.
 */
void CalcCost(Array2D<short>& costs, long flow, long arcrow, long arccol,
              long nflow, long /*nrow*/, paramT *params,
              long *poscostptr, long *negcostptr, LPCostTag /*tag*/){

  double p;
  short flow2;

  /* Lp-norm */
  flow2=flow+nflow;
  p=params->p;
  *poscostptr=LRound(costs(arcrow,arccol)*
                     (pow(labs(flow2),p)-pow(labs(flow),p)));
  flow2=flow-nflow;
  *negcostptr=LRound(costs(arcrow,arccol)*
                     (pow(labs(flow2),p)-pow(labs(flow),p)));

  /* done */
  return;
}


/* function: CalcCost()
 * ---------------------------
 * Calculates the L0 arc cost given an array of bidirectional cost weights.
 */
void CalcCost(Array2D<bidircostT>& costs, long flow, long arcrow, long arccol,
              long nflow, long /*nrow*/, paramT */*params*/,
              long *poscostptr, long *negcostptr, L0BiDirCostTag /*tag*/){
  
  long newflow, cost0;

  /* L0-norm */
  if(flow>0){
    cost0=costs(arcrow,arccol).posweight;
  }else if(flow<0){
    cost0=costs(arcrow,arccol).negweight;
  }else{
    cost0=0;
  }
  newflow=flow+nflow;
  if(newflow>0){
    *poscostptr=costs(arcrow,arccol).posweight-cost0;
  }else if(newflow<0){
    *poscostptr=costs(arcrow,arccol).negweight-cost0;
  }else{
    *poscostptr=-cost0;
  }
  newflow=flow-nflow;
  if(newflow>0){
    *negcostptr=costs(arcrow,arccol).posweight-cost0;
  }else if(newflow<0){
    *negcostptr=costs(arcrow,arccol).negweight-cost0;
  }else{
    *negcostptr=-cost0;
  }

  /* done */
  return;
}


/* function: CalcCost()
 * ---------------------------
 * Calculates the L1 arc cost given an array of bidirectional cost weights.
 */
void CalcCost(Array2D<bidircostT>& costs, long flow, long arcrow, long arccol,
              long nflow, long /*nrow*/, paramT */*params*/,
              long *poscostptr, long *negcostptr, L1BiDirCostTag /*tag*/){

  long newflow, cost0;

  /* L1-norm */
  if(flow>0){
    cost0=costs(arcrow,arccol).posweight*flow;
  }else{
    cost0=-costs(arcrow,arccol).negweight*flow;
  }
  newflow=flow+nflow;
  if(newflow>0){
    *poscostptr=(costs(arcrow,arccol).posweight*newflow
                 -cost0);
  }else{
    *poscostptr=(-costs(arcrow,arccol).negweight*newflow
                 -cost0);
  }
  newflow=flow-nflow;
  if(newflow>0){
    *negcostptr=(costs(arcrow,arccol).posweight*newflow
                 -cost0);
  }else{
    *negcostptr=(-costs(arcrow,arccol).negweight*newflow
                 -cost0);
  }

  /* done */
  return;

}


/* function: CalcCost()
 * ---------------------------
 * Calculates the L2 arc cost given an array of bidirectional cost weights.
 */
void CalcCost(Array2D<bidircostT>& costs, long flow, long arcrow, long arccol,
              long nflow, long /*nrow*/, paramT */*params*/,
              long *poscostptr, long *negcostptr, L2BiDirCostTag /*tag*/){

  long newflow, cost0;

  /* L2-norm */
  if(flow>0){
    cost0=costs(arcrow,arccol).posweight*flow*flow;
  }else{
    cost0=costs(arcrow,arccol).negweight*flow*flow;
  }
  newflow=flow+nflow;
  if(newflow>0){
    *poscostptr=(costs(arcrow,arccol).posweight
                 *newflow*newflow-cost0);
  }else{
    *poscostptr=(costs(arcrow,arccol).negweight
                 *newflow*newflow-cost0);
  }
  newflow=flow-nflow;
  if(newflow>0){
    *negcostptr=(costs(arcrow,arccol).posweight
                 *newflow*newflow-cost0);
  }else{
    *negcostptr=(costs(arcrow,arccol).negweight
                 *newflow*newflow-cost0);
  }

  /* done */
  return;
}


/* function: CalcCost()
 * ---------------------------
 * Calculates the Lp arc cost given an array of bidirectional cost weights.
 */
void CalcCost(Array2D<bidircostT>& costs, long flow, long arcrow, long arccol,
              long nflow, long /*nrow*/, paramT *params,
              long *poscostptr, long *negcostptr, LPBiDirCostTag /*tag*/){

  long newflow;
  double p, cost0;

  /* Lp-norm */
  p=params->p;
  if(flow>0){
    cost0=costs(arcrow,arccol).posweight*pow(flow,p);
  }else{
    cost0=costs(arcrow,arccol).negweight*pow(-flow,p);
  }
  newflow=flow+nflow;
  if(newflow>0){
    *poscostptr=LRound(costs(arcrow,arccol).posweight
                       *pow(newflow,p)-cost0);
  }else{
    *poscostptr=LRound(costs(arcrow,arccol).negweight
                       *pow(newflow,p)-cost0);
  }
  newflow=flow-nflow;
  if(newflow>0){
    *negcostptr=LRound(costs(arcrow,arccol).posweight
                       *pow(newflow,p)-cost0);
  }else{
    *negcostptr=LRound(costs(arcrow,arccol).negweight
                       *pow(newflow,p)-cost0);
  }

  /* done */
  return;
}


/* function: CalcCost()
 * ---------------------------
 * Calculates the arc cost given an array of long integer cost lookup tables.
 *
 * The cost array for each arc gives the cost for +/-flowmax units of
 * flow around the flow value with minimum cost, which is not
 * necessarily flow == 0.  The offset between the flow value with
 * minimum cost and flow == 0 is given by arroffset = costarr[0].
 * Positive flow values k for k = 1 to flowmax relative to this min
 * cost flow value are in costarr[k].  Negative flow values k relative
 * to the min cost flow from k = -1 to -flowmax costarr[flowmax-k].
 * costarr[2*flowmax+1] contains a scaling factor for extrapolating
 * beyond the ends of the cost table, assuming quadratically (with an offset)
 * increasing cost (subject to rounding and scaling).
 *
 * As of summer 2019, the rationale for how secondary costs are
 * extrapolated beyond the end of the table has been lost to time, but
 * the logic at least does give a self-consistent cost function that
 * is continuous at +/-flowmax and quadratically increases beyond,
 * albeit not necessarily with a starting slope that has an easily
 * intuitive basis.
 */
void CalcCost(Array2D<Array1D<long>>& costs, long flow, long arcrow, long arccol,
              long nflow, long /*nrow*/, paramT *params,
              long *poscostptr, long *negcostptr, NonGridCostTag /*tag*/){

  long xflow, flowmax, poscost, negcost, nflowsq, arroffset, sumsigsqinv;
  long abscost0;
  double c1;

  /* set up */
  flowmax=params->scndryarcflowmax;
  auto& costarr=costs(arcrow,arccol);
  arroffset=costarr[0];
  sumsigsqinv=costarr[2*flowmax+1];

  /* return zero costs if this is a zero cost arc */
  if(sumsigsqinv==ZEROCOSTARC){
    *poscostptr=0;
    *negcostptr=0;
    return;
  }

  /* compute cost of current flow */
  xflow=flow+arroffset;
  if(xflow>flowmax){
    c1=costarr[flowmax]/(double )flowmax-sumsigsqinv*flowmax;
    abscost0=(sumsigsqinv*xflow+LRound(c1))*xflow;
  }else if(xflow<-flowmax){
    c1=costarr[2*flowmax]/(double )flowmax-sumsigsqinv*flowmax;
    abscost0=(sumsigsqinv*xflow+LRound(c1))*xflow;
  }else{
    if(xflow>0){
      abscost0=costarr[xflow];
    }else if(xflow<0){
      abscost0=costarr[flowmax-xflow];  
    }else{
      abscost0=0;
    }
  }

  /* compute costs of positive and negative flow increments */
  xflow=flow+arroffset+nflow;
  if(xflow>flowmax){
    c1=costarr[flowmax]/(double )flowmax-sumsigsqinv*flowmax;    
    poscost=((sumsigsqinv*xflow+LRound(c1))*xflow)-abscost0;
  }else if(xflow<-flowmax){
    c1=costarr[2*flowmax]/(double )flowmax-sumsigsqinv*flowmax;    
    poscost=((sumsigsqinv*xflow+LRound(c1))*xflow)-abscost0;
  }else{
    if(xflow>0){
      poscost=costarr[xflow]-abscost0;
    }else if(xflow<0){
      poscost=costarr[flowmax-xflow]-abscost0;
    }else{
      poscost=-abscost0;
    }
  }
  xflow=flow+arroffset-nflow;
  if(xflow>flowmax){
    c1=costarr[flowmax]/(double )flowmax-sumsigsqinv*flowmax;    
    negcost=((sumsigsqinv*xflow+LRound(c1))*xflow)-abscost0;
  }else if(xflow<-flowmax){
    c1=costarr[2*flowmax]/(double )flowmax-sumsigsqinv*flowmax;    
    negcost=((sumsigsqinv*xflow+LRound(c1))*xflow)-abscost0;
  }else{
    if(xflow>0){
      negcost=costarr[xflow]-abscost0;
    }else if(xflow<0){
      negcost=costarr[flowmax-xflow]-abscost0;
    }else{
      negcost=-abscost0;
    }
  }

  /* scale for this flow increment and set output values */
  nflowsq=nflow*nflow;
  if(poscost>0){
    *poscostptr=(long )ceil((double )poscost/nflowsq);
  }else{
    *poscostptr=(long )floor((double )poscost/nflowsq);
  }
  if(negcost>0){
    *negcostptr=(long )ceil((double )negcost/nflowsq);
  }else{
    *negcostptr=(long )floor((double )negcost/nflowsq);
  }

  /* done */
  return;

}


/* function: EvalCost()
 * ------------------------
 * Calculates topography arc cost given an array of cost data structures.
 */
long EvalCost(Array2D<costT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, TopoCostTag /*tag*/){

  long idz1, cost1, dzmax;
  costT *cost;

  /* get arc info */
  cost=&costs(arcrow,arccol);

  /* just return 0 if we have zero cost arc */
  if(cost->sigsq==LARGESHORT){
    return(0);
  }
  
  /* compute argument to cost function */
  if(arcrow<nrow-1){

    /* row cost: dz symmetric with respect to origin */
    idz1=labs(flows(arcrow,arccol)*(params->nshortcycle)+cost->offset);
    dzmax=cost->dzmax;

  }else{

    /* column cost: non-symmetric dz */
    idz1=flows(arcrow,arccol)*(params->nshortcycle)+cost->offset;
    if((dzmax=cost->dzmax)<0){
      idz1*=-1;
      dzmax*=-1;
    }

  }

  /* calculate and return cost */
  if(idz1>dzmax){
    idz1-=dzmax;
    cost1=(idz1*idz1)/((params->layfalloffconst)*(cost->sigsq))+cost->laycost;
  }else{
    cost1=(idz1*idz1)/cost->sigsq;
    if(cost->laycost!=NOCOSTSHELF && idz1>0 && cost1>cost->laycost){
      cost1=cost->laycost;
    }
  }
  return(cost1);
}


/* function: EvalCost()
 * ------------------------
 * Calculates deformation arc cost given an array of cost data structures.
 */
long EvalCost(Array2D<costT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long /*nrow*/, paramT *params, DefoCostTag /*tag*/){

  long idz1, cost1;
  costT *cost;

  /* get arc info */
  cost=&costs(arcrow,arccol);

  /* just return 0 if we have zero cost arc */
  if(cost->sigsq==LARGESHORT){
    return(0);
  }

  /* compute argument to cost function */
  idz1=labs(flows(arcrow,arccol)*(params->nshortcycle)+cost->offset);

  /* calculate and return cost */
  if(idz1>cost->dzmax){
    idz1-=cost->dzmax;
    cost1=(idz1*idz1)/((params->layfalloffconst)*(cost->sigsq))+cost->laycost; 
  }else{
    cost1=(idz1*idz1)/cost->sigsq;
    if(cost->laycost!=NOCOSTSHELF && cost1>cost->laycost){
      cost1=cost->laycost;
    }

  }
  return(cost1);
}


/* function: EvalCost()
 * --------------------------
 * Calculates smooth-solution arc cost given an array of
 * smoothcost data structures.
 */
long EvalCost(Array2D<smoothcostT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long /*nrow*/, paramT *params, SmoothCostTag /*tag*/){

  long idz1;
  smoothcostT *cost;

  /* get arc info */
  cost=&costs(arcrow,arccol);

  /* just return 0 if we have zero cost arc */
  if(cost->sigsq==LARGESHORT){
    return(0);
  }

  /* compute argument to cost function */
  idz1=labs(flows(arcrow,arccol)*(params->nshortcycle)+cost->offset);

  /* calculate and return cost */
  return((idz1*idz1)/cost->sigsq);

}


/* function: EvalCost()
 * ----------------------
 * Calculates the L0 arc cost given an array of cost data structures.
 */
long EvalCost(Array2D<short>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long /*nrow*/, paramT * /*params*/, L0CostTag /*tag*/){

  /* L0-norm */
  if(flows(arcrow,arccol)){
    return((long)costs(arcrow,arccol));
  }else{
    return(0);
  }
}


/* function: EvalCost()
 * ----------------------
 * Calculates the L1 arc cost given an array of cost data structures.
 */
long EvalCost(Array2D<short>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long /*nrow*/, paramT * /*params*/, L1CostTag /*tag*/){

  /* L1-norm */
  return((long )((costs(arcrow,arccol))
                 *labs(flows(arcrow,arccol))));
}


/* function: EvalCost()
 * ----------------------
 * Calculates the L2 arc cost given an array of cost data structures.
 */
long EvalCost(Array2D<short>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long /*nrow*/, paramT * /*params*/, L2CostTag /*tag*/){

  /* L2-norm */
  return((long )((costs(arcrow,arccol))
                 *(flows(arcrow,arccol)*flows(arcrow,arccol))));
}


/* function: EvalCost()
 * ----------------------
 * Calculates the Lp arc cost given an array of cost data structures.
 */
long EvalCost(Array2D<short>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long /*nrow*/, paramT *params, LPCostTag /*tag*/){

  /* Lp-norm */
  return(LRound((costs(arcrow,arccol)) *
                pow(labs(flows(arcrow,arccol)),params->p)));
}


/* function: EvalCost()
 * ---------------------------
 * Calculates the L0 arc cost given an array of bidirectional cost structures.
 */
long EvalCost(Array2D<bidircostT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long /*nrow*/, paramT * /*params*/, L0BiDirCostTag /*tag*/){

  /* L0-norm */
  if(flows(arcrow,arccol)>0){
    return((long )costs(arcrow,arccol).posweight);
  }else if(flows(arcrow,arccol)<0){
    return((long )costs(arcrow,arccol).negweight);
  }else{
    return(0);
  }
}


/* function: EvalCost()
 * ---------------------------
 * Calculates the L1 arc cost given an array of bidirectional cost structures.
 */
long EvalCost(Array2D<bidircostT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long /*nrow*/, paramT * /*params*/, L1BiDirCostTag /*tag*/){

  /* L1-norm */
  if(flows(arcrow,arccol)>0){
    return((long )((costs(arcrow,arccol).posweight)
                   *(flows(arcrow,arccol))));
  }else{
    return((long )((costs(arcrow,arccol).negweight)
                   *(-flows(arcrow,arccol))));
  }
}


/* function: EvalCost()
 * ---------------------------
 * Calculates the L2 arc cost given an array of bidirectional cost structures.
 */
long EvalCost(Array2D<bidircostT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long /*nrow*/, paramT * /*params*/, L2BiDirCostTag /*tag*/){

  /* L2-norm */
  if(flows(arcrow,arccol)>0){
    return((long )((costs(arcrow,arccol).posweight)
                   *(flows(arcrow,arccol)*flows(arcrow,arccol))));
  }else{
    return((long )((costs(arcrow,arccol).negweight)
                   *(flows(arcrow,arccol)*flows(arcrow,arccol))));
  }
}


/* function: EvalCost()
 * ---------------------------
 * Calculates the Lp arc cost given an array of bidirectional cost structures.
 */
long EvalCost(Array2D<bidircostT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long /*nrow*/, paramT *params, LPBiDirCostTag /*tag*/){

  /* Lp-norm */
  if(flows(arcrow,arccol)>0){
    return(LRound((costs(arcrow,arccol).posweight)
                  *pow(flows(arcrow,arccol),params->p)));
  }else{
    return(LRound((costs(arcrow,arccol).posweight)
                  *pow(-flows(arcrow,arccol),params->p)));
  }
}


/* function: EvalCost()
 * ---------------------------
 * Calculates the arc cost given an array of long integer cost lookup tables.
 */
long EvalCost(Array2D<Array1D<long>>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long /*nrow*/, paramT *params, NonGridCostTag /*tag*/){

  long flow, xflow, flowmax, arroffset, sumsigsqinv;
  double c1;

  /* set up */
  flow=flows(arcrow,arccol);
  flowmax=params->scndryarcflowmax;
  auto& costarr=costs(arcrow,arccol);
  arroffset=costarr[0];
  sumsigsqinv=costarr[2*flowmax+1];

  /* return zero costs if this is a zero cost arc */
  if(sumsigsqinv==ZEROCOSTARC){
    return(0);
  }

  /* compute cost of current flow */
  xflow=flow+arroffset;
  if(xflow>flowmax){
    c1=costarr[flowmax]/(double )flowmax-sumsigsqinv*flowmax;
    return((sumsigsqinv*xflow+LRound(c1))*xflow);
  }else if(xflow<-flowmax){
    c1=costarr[2*flowmax]/(double )flowmax-sumsigsqinv*flowmax;
    return((sumsigsqinv*xflow+LRound(c1))*xflow);
  }else{
    if(xflow>0){
      return(costarr[xflow]);
    }else if(xflow<0){
      return(costarr[flowmax-xflow]);
    }else{
      return(0);
    }
  }
}


/* function: CalcInitMaxFlow()
 * ---------------------------
 * Calculates the maximum flow magnitude to allow in the initialization
 * by examining the dzmax members of arc statistical cost data structures.
 */
static
int CalcInitMaxFlow(paramT *params, Array2D<costT>& costs, long nrow, long ncol){

  long row, col, maxcol, initmaxflow, arcmaxflow;

  if(params->initmaxflow<=0){
    if(params->costmode==NOSTATCOSTS){
      params->initmaxflow=NOSTATINITMAXFLOW;
    }else{
      if(params->costmode==TOPO || params->costmode==DEFO){
        initmaxflow=0;
        for(row=0;row<2*nrow-1;row++){
          if(row<nrow-1){
            maxcol=ncol;
          }else{
            maxcol=ncol-1;
          }
          for(col=0;col<maxcol;col++){
            if(costs(row,col).dzmax!=LARGESHORT){
              arcmaxflow=ceil(labs((long )costs(row,col).dzmax)/
                              (double )(params->nshortcycle)
                              +params->arcmaxflowconst);
              if(arcmaxflow>initmaxflow){
                initmaxflow=arcmaxflow;
              }
            }
          }
        }
        params->initmaxflow=initmaxflow;
      }else{
        params->initmaxflow=DEF_INITMAXFLOW;
      }
    }
  }
  return(0);
}


#define INSTANTIATE_TEMPLATES(T) \
  template int BuildCostArrays(Array2D<typename T::Cost>*, Array2D<short>*, \
                               Array2D<float>&, Array2D<float>&, \
                               Array2D<float>&, long, long, \
                               long, long, paramT*, \
                               tileparamT*, infileT*, \
                               outfileT*, T);
INSTANTIATE_TEMPLATES(TopoCostTag)
INSTANTIATE_TEMPLATES(DefoCostTag)
INSTANTIATE_TEMPLATES(SmoothCostTag)
INSTANTIATE_TEMPLATES(L0CostTag)
INSTANTIATE_TEMPLATES(L1CostTag)
INSTANTIATE_TEMPLATES(L2CostTag)
INSTANTIATE_TEMPLATES(LPCostTag)
INSTANTIATE_TEMPLATES(L0BiDirCostTag)
INSTANTIATE_TEMPLATES(L1BiDirCostTag)
INSTANTIATE_TEMPLATES(L2BiDirCostTag)
INSTANTIATE_TEMPLATES(LPBiDirCostTag)
INSTANTIATE_TEMPLATES(NonGridCostTag)

} // namespace isce3::unwrap
