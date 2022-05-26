/*************************************************************************

  snaphu tile-mode source file
  Written by Curtis W. Chen
  Copyright 2002 Board of Trustees, Leland Stanford Jr. University
  Please see the supporting documentation for terms of use.
  No warranty.

*************************************************************************/

#include <cmath>
#include <csignal>
#include <cstring>
#include <type_traits>
#include <unistd.h>

#include <isce3/except/Error.h>

#include "snaphu.h"

namespace isce3::unwrap {

/* static (local) function prototypes */
static
long ThickenCosts(Array2D<incrcostT>& incrcosts, long nrow, long ncol);
static
nodeT *RegionsNeighborNode(nodeT *node1, long *arcnumptr, Array2D<nodeT>& nodes,
                           long *arcrowptr, long *arccolptr,
                           long nrow, long ncol);
static
int ClearBuckets(bucketT *bkts);
static
int MergeRegions(Array2D<nodeT>& nodes, nodeT *source, Array1D<long>& regionsizes,
                  long closestregion, long nrow, long ncol);
static
int RenumberRegion(Array2D<nodeT>& nodes, nodeT *source, long newnum,
                   long nrow, long ncol);
template<class Cost>
static
int ReadNextRegion(long tilerow, long tilecol, long nlines, long linelen,
                   outfileT *outfiles, paramT *params,
                   Array2D<short>* nextregionsptr, Array2D<float>* nextunwphaseptr,
                   Array2D<Cost>* nextcostsptr,
                   long *nextnrowptr, long *nextncolptr);
static
int SetTileReadParams(tileparamT *tileparams, long nexttilenlines, 
                      long nexttilelinelen, long tilerow, long tilecol, 
                      long nlines, long linelen, paramT *params);
template<class Cost>
static
int ReadEdgesAboveAndBelow(long tilerow, long tilecol, long nlines,
                           long linelen, paramT *params, outfileT *outfiles,
                           Array2D<short>& regionsabove, Array2D<short>& regionsbelow,
                           Array2D<float>& unwphaseabove, Array2D<float>& unwphasebelow,
                           Array2D<Cost>& costsabove, Array2D<Cost>& costsbelow);
template<class CostTag>
static
int TraceRegions(Array2D<short>& regions, Array2D<short>& nextregions, 
                 Array2D<short>& lastregions, Array2D<short>& regionsabove, 
                 Array2D<short>& regionsbelow, Array2D<float>& unwphase,
                 Array2D<float>& nextunwphase, Array2D<float>& lastunwphase,
                 Array2D<float>& unwphaseabove, Array2D<float>& unwphasebelow,
                 Array2D<typename CostTag::Cost>& costs, 
                 Array2D<typename CostTag::Cost>& nextcosts,
                 Array2D<typename CostTag::Cost>& lastcosts, 
                 Array2D<typename CostTag::Cost>& costsabove,
                 Array2D<typename CostTag::Cost>& costsbelow, long prevnrow, long prevncol, 
                 long tilerow, long tilecol, long nrow, long ncol, Array2D<nodeT>& scndrynodes,
                 Array2D<nodesuppT>& nodesupp, Array2D<scndryarcT>& scndryarcs,
                 Array2D<Array1D<long>>& scndrycosts, Array1D<int>& nscndrynodes,
                 Array1D<int>& nscndryarcs, Array1D<long>& totarclens, 
                 Array2D<short>& bulkoffsets, paramT *params, CostTag tag);
static
long FindNumPathsOut(nodeT *from, paramT *params, long tilerow, long tilecol,
                     long nnrow, long nncol, Array2D<short>& regions,
                     Array2D<short>& nextregions, Array2D<short>& lastregions,
                     Array2D<short>& regionsabove, Array2D<short>& regionsbelow, long prevncol);
template<class CostTag>
static
int RegionTraceCheckNeighbors(nodeT *from, nodeT **nextnodeptr,
                              Array2D<nodeT>& primarynodes, Array2D<short>& regions,
                              Array2D<short>& nextregions, Array2D<short>& lastregions,
                              Array2D<short>& regionsabove, Array2D<short>& regionsbelow,
                              long tilerow, long tilecol, long nnrow,
                              long nncol, Array2D<nodeT>& scndrynodes,
                              Array2D<nodesuppT>& nodesupp, Array2D<scndryarcT>& scndryarcs,
                              long *nnewnodesptr, long *nnewarcsptr,
                              long flowmax, long nrow, long ncol,
                              long prevnrow, long prevncol, paramT *params,
                              Array2D<typename CostTag::Cost>& costs, 
                              Array2D<typename CostTag::Cost>& rightedgecosts,
                              Array2D<typename CostTag::Cost>& loweredgecosts, 
                              Array2D<typename CostTag::Cost>& leftedgecosts,
                              Array2D<typename CostTag::Cost>& upperedgecosts, 
                              Array2D<short>& flows, Array2D<short>& rightedgeflows, 
                              Array2D<short>& loweredgeflows, Array2D<short>& leftedgeflows, 
                              Array2D<short>& upperedgeflows, Array2D<Array1D<long>>& scndrycosts,
                              Array1D<nodeT*>* updatednontilenodesptr,
                              long *nupdatednontilenodesptr,
                              long *updatednontilenodesizeptr,
                              Array1D<short>* inontilenodeoutarcptr,
                              long *totarclenptr, CostTag tag);
template<class Cost>
static
int SetUpperEdge(long ncol, long tilerow, long tilecol, Array2D<Cost>& costs,
                 Array2D<Cost>& costsabove, Array2D<float>& unwphase,
                 Array2D<float>& unwphaseabove, Array2D<Cost>& upperedgecosts,
                 Array2D<short>& upperedgeflows, paramT *params, Array2D<short>& bulkoffsets);
template<class Cost>
static
int SetLowerEdge(long nrow, long ncol, long tilerow, long tilecol,
                 Array2D<Cost>& costs, Array2D<Cost>& costsbelow,
                 Array2D<float>& unwphase, Array2D<float>& unwphasebelow,
                 Array2D<Cost>& loweredgecosts, Array2D<short>& loweredgeflows,
                 paramT *params, Array2D<short>& bulkoffsets);
template<class Cost>
static
int SetLeftEdge(long nrow, long prevncol, long tilerow, long tilecol,
                Array2D<Cost>& costs, Array2D<Cost>& lastcosts, Array2D<float>& unwphase,
                Array2D<float>& lastunwphase, Array2D<Cost>& leftedgecosts,
                Array2D<short>& leftedgeflows, paramT *params, Array2D<short>& bulkoffsets);
template<class Cost>
static
int SetRightEdge(long nrow, long ncol, long tilerow, long tilecol,
                 Array2D<Cost>& costs, Array2D<Cost>& nextcosts,
                 Array2D<float>& unwphase, Array2D<float>& nextunwphase,
                 Array2D<Cost>& rightedgecosts, Array2D<short>& rightedgeflows,
                 paramT *params, Array2D<short>& bulkoffsets);
static
short AvgSigSq(short sigsq1, short sigsq2);
template<class CostTag>
static
int TraceSecondaryArc(nodeT *primaryhead, Array2D<nodeT>& scndrynodes,
                      Array2D<nodesuppT>& nodesupp, Array2D<scndryarcT>& scndryarcs,
                      Array2D<Array1D<long>>& scndrycosts, long *nnewnodesptr,
                      long *nnewarcsptr, long tilerow, long tilecol,
                      long flowmax, long nrow, long ncol,
                      long prevnrow, long prevncol, paramT *params,
                      Array2D<typename CostTag::Cost>& tilecosts, 
                      Array2D<typename CostTag::Cost>& rightedgecosts,
                      Array2D<typename CostTag::Cost>& loweredgecosts, 
                      Array2D<typename CostTag::Cost>& leftedgecosts,
                      Array2D<typename CostTag::Cost>& upperedgecosts, 
                      Array2D<short>& tileflows, Array2D<short>& rightedgeflows, 
                      Array2D<short>& loweredgeflows, Array2D<short>& leftedgeflows, 
                      Array2D<short>& upperedgeflows, Array1D<nodeT*>* updatednontilenodesptr,
                      long *nupdatednontilenodesptr, long *updatednontilenodesizeptr,
                      Array1D<short>* inontilenodeoutarcptr, long *totarclenptr, CostTag tag);
static
nodeT *FindScndryNode(Array2D<nodeT>& scndrynodes, Array2D<nodesuppT>& nodesupp,
                      long tilenum, long primaryrow, long primarycol);
static
int IntegrateSecondaryFlows(long linelen, long nlines, Array2D<nodeT>& scndrynodes,
                            Array2D<nodesuppT>& nodesupp, Array2D<scndryarcT>& scndryarcs,
                            Array1D<int>& nscndryarcs, Array2D<short>& scndryflows,
                            Array2D<short>& bulkoffsets, outfileT *outfiles,
                            paramT *params);
static
int ParseSecondaryFlows(long tilenum, Array1D<int>& nscndryarcs, Array2D<short>& tileflows,
                        Array2D<short>& regions, Array2D<short>& scndryflows,
                        Array2D<nodesuppT>& nodesupp, Array2D<scndryarcT>& scndryarcs,
                        long nrow, long ncol, long ntilerow, long ntilecol,
                        paramT *params);
static
int AssembleTileConnComps(long linelen, long nlines,
                          outfileT *outfiles, paramT *params);
static
int ConnCompSizeNPixCompare(const void *ptr1, const void *ptr2);




/* function: SetupTile()
 * ---------------------
 * Sets up tile parameters and output file names for the current tile.
 */
int SetupTile(long nlines, long linelen, paramT *params, 
              tileparamT *tileparams, outfileT *outfiles, 
              outfileT *tileoutfiles, long tilerow, long tilecol){

  long ni, nj;
  char path[MAXSTRLEN]={}, basename[MAXSTRLEN]={};
  char *tiledir;
  std::string tempstring;

  /* set parameters for current tile */
  ni=ceil((nlines+(params->ntilerow-1)*params->rowovrlp)
          /(double )params->ntilerow);
  nj=ceil((linelen+(params->ntilecol-1)*params->colovrlp)
          /(double )params->ntilecol);
  tileparams->firstrow=tilerow*(ni-params->rowovrlp);
  tileparams->firstcol=tilecol*(nj-params->colovrlp);
  if(tilerow==params->ntilerow-1){
    tileparams->nrow=nlines-(params->ntilerow-1)*(ni-params->rowovrlp);
  }else{
    tileparams->nrow=ni;
  }
  if(tilecol==params->ntilecol-1){
    tileparams->ncol=linelen-(params->ntilecol-1)*(nj-params->colovrlp);
  }else{
    tileparams->ncol=nj;
  }

  /* error checking on tile size */
  if(params->minregionsize > (tileparams->nrow)*(tileparams->ncol)){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Minimum region size cannot exceed tile size");
  }

  /* set output files */
  tiledir=params->tiledir;
  ParseFilename(outfiles->outfile,path,basename);
  tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
    +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
      +std::to_string(tileparams->ncol);
  StrNCopy(tileoutfiles->outfile,tempstring.c_str(),MAXSTRLEN);
  if(strlen(outfiles->initfile)){
    ParseFilename(outfiles->initfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->initfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->initfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->flowfile)){
    ParseFilename(outfiles->flowfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->flowfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->flowfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->eifile)){
    ParseFilename(outfiles->eifile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->eifile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->eifile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->rowcostfile)){
    ParseFilename(outfiles->rowcostfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->rowcostfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->rowcostfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->colcostfile)){
    ParseFilename(outfiles->colcostfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->colcostfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->colcostfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->mstrowcostfile)){
    ParseFilename(outfiles->mstrowcostfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->mstrowcostfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->mstrowcostfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->mstcolcostfile)){
    ParseFilename(outfiles->mstcolcostfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->mstcolcostfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->mstcolcostfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->mstcostsfile)){
    ParseFilename(outfiles->mstcostsfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->mstcostsfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->mstcostsfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->corrdumpfile)){
    ParseFilename(outfiles->corrdumpfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->corrdumpfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->corrdumpfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->rawcorrdumpfile)){
    ParseFilename(outfiles->rawcorrdumpfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->rawcorrdumpfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->rawcorrdumpfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->conncompfile)){
    ParseFilename(outfiles->conncompfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->conncompfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->conncompfile,"",MAXSTRLEN);
  }
  if(strlen(outfiles->costoutfile)){
    ParseFilename(outfiles->costoutfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->costoutfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+TMPTILECOSTSUFFIX
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->costoutfile,tempstring.c_str(),MAXSTRLEN);
  }
  if(strlen(outfiles->logfile)){
    ParseFilename(outfiles->logfile,path,basename);
    tempstring=std::string(tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
        +std::to_string(tileparams->ncol);
    StrNCopy(tileoutfiles->logfile,tempstring.c_str(),MAXSTRLEN);
  }else{
    StrNCopy(tileoutfiles->logfile,"",MAXSTRLEN);
  }
  tileoutfiles->outfileformat=TMPTILEOUTFORMAT;

  /* done */
  return(0);

}


/* function: SetUpDoTileMask()
 * ---------------------------
 * Read the tile mask if a file name is specified in the infiles structure,
 * otherwise return an array of all ones.
 */
Array2D<signed char>
SetUpDoTileMask(infileT *infiles, long ntilerow, long ntilecol){

  long row, col;
  tileparamT readparams[1]={};

  /* get memory */
  auto dotilemask = Array2D<signed char>(ntilerow, ntilecol);

  /* see if a file name was passed */
  if(strlen(infiles->dotilemaskfile)){

    /* read the input file */
    readparams->nrow=ntilerow;
    readparams->ncol=ntilecol;
    readparams->firstrow=0;
    readparams->firstcol=0;
    Read2DArray(&dotilemask,infiles->dotilemaskfile,ntilecol,
                ntilerow,readparams,sizeof(signed char *),sizeof(signed char));

  }else{

    /* set array to be all ones */
    for(row=0;row<ntilerow;row++){
      for(col=0;col<ntilecol;col++){
        dotilemask(row,col)=1;
      }
    }
  }

  /* return the array pointer */
  return(dotilemask);

}


/* function: GrowRegions()
 * -----------------------
 * Grows contiguous regions demarcated by arcs whose residual costs are
 * less than some threshold.  Numbers the regions sequentially from 0.
 */
template<class CostTag>
int GrowRegions(Array2D<typename CostTag::Cost>& costs, Array2D<short>& flows, long nrow, long ncol,
                Array2D<incrcostT>& incrcosts, outfileT *outfiles,
                tileparamT * /*tileparams*/, paramT *params, CostTag tag){

  long i, row, col, maxcol;
  long arcrow, arccol, arcnum, fromdist, arcdist;
  long regioncounter, regionsizeslen, *thisregionsize;
  long closestregiondist, closestregion, lastfromdist;
  long costthresh, minsize, maxcost;
  long costtypesize;
  nodeT *source, *from, *to, *ground;
  bucketT bkts[1]={};
  tileparamT temptileparams[1]={};

  using Cost=typename CostTag::Cost;
  Array2D<Cost> growregionscosts;
  Array2D<Cost>* growregionscostsptr=&growregionscosts;

  constexpr int output_detail_level=2;
  auto verbose=pyre::journal::info_t("isce3.unwrap.snaphu",output_detail_level);

  /* set up */
  auto info=pyre::journal::info_t("isce3.unwrap.snaphu");
  info << pyre::journal::at(__HERE__)
       << "Growing reliable regions"
       << pyre::journal::endl;
  minsize=params->minregionsize;
  costthresh=params->tilecostthresh;
  closestregion=0;

  /* reread statistical costs from stored file if costs array is for Lp mode */
  if(params->p >= 0){
    if(params->costmode==TOPO){
      costtypesize=sizeof(costT);
    }else if(params->costmode==DEFO){
      costtypesize=sizeof(costT);
    }else if(params->costmode==SMOOTH){
      costtypesize=sizeof(smoothcostT);
    }else{
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Illegal cost mode in GrowRegions(). This is a bug.");
    }
    temptileparams->firstrow=0;
    temptileparams->firstcol=0;
    temptileparams->nrow=nrow;
    temptileparams->ncol=ncol;
    Read2DRowColFile(growregionscostsptr,outfiles->costoutfile,
                     ncol,nrow,temptileparams,costtypesize);
  }else{
    *growregionscostsptr=costs;
  }

  /* loop over all arcs */
  for(arcrow=0;arcrow<2*nrow-1;arcrow++){
    if(arcrow<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(arccol=0;arccol<maxcol;arccol++){

      /* compute incremental costs of unit flows in either direction */
      ReCalcCost(*growregionscostsptr,incrcosts,flows(arcrow,arccol),
                 arcrow,arccol,1,nrow,params,tag);

      /* store lesser of incremental costs in first field */
      if(incrcosts(arcrow,arccol).negcost<incrcosts(arcrow,arccol).poscost){
        incrcosts(arcrow,arccol).poscost=incrcosts(arcrow,arccol).negcost;
      }

      /* subtract costthresh and take negative of costs, then clip to zero */
      incrcosts(arcrow,arccol).poscost
        =-(incrcosts(arcrow,arccol).poscost-costthresh);
      if(incrcosts(arcrow,arccol).poscost<0){
        incrcosts(arcrow,arccol).poscost=0;
      }
    }
  }

  /* thicken the costs arrays; results stored in negcost field */
  maxcost=ThickenCosts(incrcosts,nrow,ncol);

  /* initialize nodes and buckets for region growing */
  ground=NULL;
  auto nodes=Array2D<nodeT>(nrow,ncol);
  InitNodeNums(nrow,ncol,nodes,ground);
  InitNodes(nrow,ncol,nodes,ground);
  bkts->size=maxcost+2;
  bkts->minind=0;
  bkts->maxind=bkts->size-1;
  bkts->curr=0;
  bkts->wrapped=FALSE;
  bkts->bucketbase=Array1D<nodeT*>(bkts->size);
  bkts->bucket=bkts->bucketbase.data();
  for(i=0;i<bkts->size;i++){
    bkts->bucket[i]=NULL;
  }

  /* initialize region variables */
  regioncounter=-1;
  regionsizeslen=INITARRSIZE;
  auto regionsizes=Array1D<long>(regionsizeslen);
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      nodes(row,col).incost=-1;
    }
  }

  /* loop over all nodes (pixels) to make sure each is in a group */
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){

      /* see if node is not in a group */
      if(nodes(row,col).incost<0){

        /* clear the buckets */
        ClearBuckets(bkts);

        /* make node source and put it in the first bucket */
        source=&nodes(row,col);
        source->next=NULL;
        source->prev=NULL;
        source->group=INBUCKET;
        source->outcost=0;
        bkts->bucket[0]=source;
        bkts->curr=0;
        lastfromdist=0;

        /* increment the region counter */
        if(++regioncounter>=regionsizeslen){
          regionsizeslen+=INITARRSIZE;
          regionsizes.conservativeResize(regionsizeslen);
        }
        thisregionsize=&regionsizes[regioncounter];

        /* set up */
        (*thisregionsize)=0;
        closestregiondist=VERYFAR;

        /* loop to grow region */
        while(TRUE){

          /* set from node to closest node in circular bucket structure */
          from=ClosestNode(bkts);
          
          /* break if we can't grow any more and the region is big enough */
          if(from==NULL){
            if(*thisregionsize>=minsize){

              /* no more nonregion nodes, and current region is big enough */
              break;

            }else{

              /* no more nonregion nodes, but current region still too small */
              /* merge with another region */
              MergeRegions(nodes,source,regionsizes,closestregion,nrow,ncol);
              regioncounter--;
              break;

            }
          }else{
            fromdist=from->outcost;
            if(fromdist>lastfromdist){
              if(regionsizes[regioncounter]>=minsize){

                /* region grown to all nodes within mincost, is big enough */
                break;

              }
              if(fromdist>closestregiondist){

                /* another region closer than new node, so merge regions */
                MergeRegions(nodes,source,regionsizes,closestregion,nrow,ncol);
                regioncounter--;
                break;
              }
            }
          }

          /* make from node a part of the current region */
          from->incost=regioncounter;
          (*thisregionsize)++;
          lastfromdist=fromdist;

          /* scan from's neighbors */
          arcnum=0;
          while((to=RegionsNeighborNode(from,&arcnum,nodes,
                                        &arcrow,&arccol,nrow,ncol))!=NULL){

            /* get cost of arc to the to node */
            arcdist=incrcosts(arcrow,arccol).negcost;

            /* see if to node is already in another region */
            if(to->incost>=0){

              /* keep track of which neighboring region is closest */
              if(to->incost!=regioncounter && arcdist<closestregiondist){
                closestregiondist=arcdist;
                closestregion=to->incost;
              }

            }else{

              /* to node is not in another region */
              /* compare distance of new nodes to temp labels */
              if(arcdist<(to->outcost)){

                /* if to node is already in a (circular) bucket, remove it */
                if(to->group==INBUCKET){
                  BucketRemove(to,to->outcost,bkts);
                }
                
                /* update to node */
                to->outcost=arcdist;
                to->pred=from;

                /* insert to node into appropriate (circular) bucket */
                BucketInsert(to,arcdist,bkts);
                if(arcdist<bkts->curr){
                  bkts->curr=arcdist;
                }
              }
            }
          }
        }
      }
    }
  }
  verbose << pyre::journal::at(__HERE__)
          << "Tile partitioned into " << (regioncounter+1) << " regions"
          << pyre::journal::endl;

  /* write regions array */
  /* write as shorts if multiple tiles */
  if(params->ntilerow>1 || params->ntilecol>1){
    auto regions=Array2D<short>(nrow,ncol);
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        if(nodes(row,col).incost>LARGESHORT){
          fflush(NULL);
          throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                  "Number of regions in tile exceeds max allowed");
        }
        regions(row,col)=nodes(row,col).incost;
      }
    }
    auto regionfile=std::string(outfiles->outfile)+REGIONSUFFIX;
    verbose << pyre::journal::at(__HERE__)
            << "Writing region data to file " << regionfile
            << pyre::journal::endl;
    Write2DArray(regions,regionfile.c_str(),nrow,ncol,sizeof(short));
  }

  /* done */
  return(0);

}


/* function: GrowConnCompsMask()
 * -----------------------------
 * Grows contiguous regions demarcated by arcs whose residual costs are
 * less than some threshold.  Numbers the regions sequentially from 1.
 * Writes out byte file of connected component mask, with 0 for any pixels
 * not assigned to a component.
 */
template<class CostTag>
int GrowConnCompsMask(Array2D<typename CostTag::Cost>& costs, Array2D<short>& flows, long nrow, long ncol,
                      Array2D<incrcostT>& incrcosts, outfileT *outfiles,
                      paramT *params, CostTag tag){

  long i, row, col, maxcol;
  long arcrow, arccol, arcnum;
  long regioncounter, regionsizeslen, *thisregionsize;
  long costthresh, minsize, maxncomps, ntied, newnum;
  unsigned long outtypemax, outtypesize;
  nodeT *source, *from, *to, *ground;
  void *outbufptr;
  bucketT bkts[1]={};
  char realoutfile[MAXSTRLEN]={};
  FILE *conncompfp;

  constexpr int output_detail_level=2;
  auto verbose=pyre::journal::info_t("isce3.unwrap.snaphu",output_detail_level);

  /* error checking */
  auto info=pyre::journal::info_t("isce3.unwrap.snaphu");
  info << pyre::journal::at(__HERE__)
       << "Growing connected component mask"
       << pyre::journal::endl;
  minsize=params->minconncompfrac*nrow*ncol;
  maxncomps=params->maxncomps;
  costthresh=params->conncompthresh;
  if(minsize>nrow*ncol){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Minimum region size cannot exceed tile size");
  }

  /* loop over all arcs */
  for(arcrow=0;arcrow<2*nrow-1;arcrow++){
    if(arcrow<nrow-1){
      maxcol=ncol;
    }else{
      maxcol=ncol-1;
    }
    for(arccol=0;arccol<maxcol;arccol++){

      /* compute incremental costs of unit flows in either direction */
      ReCalcCost(costs,incrcosts,flows(arcrow,arccol),
                 arcrow,arccol,1,nrow,params,tag);

      /* store lesser of incremental costs in first field */
      if(incrcosts(arcrow,arccol).negcost<incrcosts(arcrow,arccol).poscost){
        incrcosts(arcrow,arccol).poscost=incrcosts(arcrow,arccol).negcost;
      }

      /* subtract costthresh and take negative of costs, then clip to zero */
      incrcosts(arcrow,arccol).poscost
        =-(incrcosts(arcrow,arccol).poscost-costthresh);
      if(incrcosts(arcrow,arccol).poscost<0){
        incrcosts(arcrow,arccol).poscost=0;
      }
    }
  }

  /* thicken the costs arrays; results stored in negcost field */
  ThickenCosts(incrcosts,nrow,ncol);

  /* initialize nodes and buckets for region growing */
  ground=NULL;
  auto nodes=Array2D<nodeT>(nrow,ncol);
  InitNodeNums(nrow,ncol,nodes,ground);
  InitNodes(nrow,ncol,nodes,ground);
  bkts->size=1;
  bkts->minind=0;
  bkts->maxind=0;
  bkts->wrapped=FALSE;
  bkts->bucketbase=Array1D<nodeT*>(1);
  bkts->bucket=bkts->bucketbase.data();
  bkts->bucket[0]=NULL;

  /* initialize region variables */
  regioncounter=0;
  regionsizeslen=INITARRSIZE;
  auto regionsizes=Array1D<long>(regionsizeslen);
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      nodes(row,col).incost=-1;
    }
  }

  /* loop over all nodes (pixels) to make sure each is in a group */
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){

      /* see if node is not in a group */
      if(nodes(row,col).incost<0){

        /* clear the buckets */
        ClearBuckets(bkts);

        /* make node source and put it in the first bucket */
        source=&nodes(row,col);
        source->next=NULL;
        source->prev=NULL;
        source->group=INBUCKET;
        source->outcost=0;
        bkts->bucket[0]=source;
        bkts->curr=0;

        /* increment the region counter */
        if(++regioncounter>=regionsizeslen){
          regionsizeslen+=INITARRSIZE;
          regionsizes.conservativeResize(regionsizeslen);
        }
        thisregionsize=&regionsizes[regioncounter];

        /* set up */
        (*thisregionsize)=0;

        /* loop to grow region */
        while(TRUE){

          /* set from node to closest node in circular bucket structure */
          from=ClosestNode(bkts);
          
          /* break if we can't grow any more and the region is big enough */
          if(from==NULL){
            if(regionsizes[regioncounter]>=minsize){

              /* no more nonregion nodes, and current region is big enough */
              break;

            }else{

              /* no more nonregion nodes, but current region still too small */
              /* zero out the region */
              RenumberRegion(nodes,source,0,nrow,ncol);
              regioncounter--;
              break;

            }
          }

          /* make from node a part of the current region */
          from->incost=regioncounter;
          (*thisregionsize)++;

          /* scan from's neighbors */
          arcnum=0;
          while((to=RegionsNeighborNode(from,&arcnum,nodes,
                                        &arcrow,&arccol,nrow,ncol))!=NULL){

            /* see if to can be reached */
            if(to->incost<0 && incrcosts(arcrow,arccol).negcost==0
               && to->group!=INBUCKET){

              /* update to node */
              to->pred=from;
              BucketInsert(to,0,bkts);

            }
          }
        }
      }
    }
  }
  verbose << pyre::journal::at(__HERE__)
          << regioncounter << " connected components formed"
          << pyre::journal::endl;

  /* make sure we don't have too many components */
  if(regioncounter>maxncomps){

    /* copy regionsizes array and sort to find new minimum region size */
    verbose << pyre::journal::at(__HERE__)
            << "Keeping only " << maxncomps << " connected components"
            << pyre::journal::endl;
    auto sortedregionsizes=Array1D<long>(regioncounter);
    for(i=0;i<regioncounter;i++){
      sortedregionsizes[i]=regionsizes[i+1];
    }
    qsort((void *)sortedregionsizes.data(),regioncounter,sizeof(long),LongCompare);
    minsize=sortedregionsizes[regioncounter-maxncomps];

    /* see how many regions of size minsize still need zeroing */
    ntied=0;
    i=regioncounter-maxncomps-1;
    while(i>=0 && sortedregionsizes[i]==minsize){
      ntied++;
      i--;
    }

    /* zero out regions that are too small */
    newnum=-1;
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        i=nodes(row,col).incost;
        if(i>0){
          if(regionsizes[i]<minsize
             || (regionsizes[i]==minsize && (ntied--)>0)){

            /* region too small, so zero it out */
            RenumberRegion(nodes,&(nodes(row,col)),0,nrow,ncol);

          }else{

            /* keep region, assign it new region number */
            /* temporarily assign negative of new number to avoid collisions */
            RenumberRegion(nodes,&(nodes(row,col)),newnum--,nrow,ncol);

          }
        }
      }
    }

    /* flip temporary negative region numbers so they are positive */
    for(row=0;row<nrow;row++){
      for(col=0;col<ncol;col++){
        nodes(row,col).incost=-nodes(row,col).incost;
      }
    }
  }

  /* write connected components as appropriate data type */
  auto ucharbuf=Array1D<unsigned char>(ncol);
  auto uintbuf=Array1D<unsigned>(ncol);
  if(params->conncompouttype==CONNCOMPOUTTYPEUCHAR){
    outtypemax=UCHAR_MAX;
    outtypesize=(int )sizeof(unsigned char);
    outbufptr=(void *)ucharbuf.data();
  }else if(params->conncompouttype==CONNCOMPOUTTYPEUINT){
    outtypemax=UINT_MAX;
    outtypesize=(int )sizeof(unsigned int);
    outbufptr=(void *)uintbuf.data();
  }else{
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Bad conncompouttype in GrowConnCompMask()");
  }
  info << pyre::journal::at(__HERE__)
       << "Writing connected components to file " << outfiles->conncompfile
       << " as " << ((int )outtypesize) << "-byte unsigned ints"
       << pyre::journal::endl;
  conncompfp=OpenOutputFile(outfiles->conncompfile,realoutfile);
  for(row=0;row<nrow;row++){
    for(col=0;col<ncol;col++){
      if(nodes(row,col).incost>outtypemax){
        fflush(NULL);
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Number of connected components too large for output type");
      }
      uintbuf[col]=(unsigned int)(nodes(row,col).incost);
    }
    if(params->conncompouttype==CONNCOMPOUTTYPEUCHAR){
      for(col=0;col<ncol;col++){
        ucharbuf[col]=(unsigned char )uintbuf[col];
      }
    }
    if(fwrite(outbufptr,outtypesize,ncol,conncompfp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while writing to file " + std::string(realoutfile) +
              " (device full?)");
    }
  }
  if(fclose(conncompfp)){
    fflush(NULL);
    auto warnings=pyre::journal::warning_t("isce3.unwrap.snaphu");
    warnings << pyre::journal::at(__HERE__)
             << "WARNING: problem closing file " << outfiles->conncompfile
             << " (disk full?)"
             << pyre::journal::endl;
  }

  return(0);

}


/* function: ThickenCosts()
 * ------------------------
 */
static
long ThickenCosts(Array2D<incrcostT>& incrcosts, long nrow, long ncol){

  long row, col, templong, maxcost;
  double n;

  auto warnings=pyre::journal::warning_t("isce3.unwrap.snaphu");

  /* initialize variable storing maximum cost */
  maxcost=-LARGEINT;

  /* loop over row arcs and convolve */
  for(row=0;row<nrow-1;row++){
    for(col=0;col<ncol;col++){
      templong=2*incrcosts(row,col).poscost;
      n=2.0;
      if(col!=0){
        templong+=incrcosts(row,col-1).poscost;
        n+=1.0;
      }
      if(col!=ncol-1){
        templong+=incrcosts(row,col+1).poscost;
        n+=1.0;
      }
      templong=LRound(templong/n);
      if(templong>LARGESHORT){
        fflush(NULL);
        warnings << pyre::journal::at(__HERE__)
                 << "WARNING: COSTS CLIPPED IN ThickenCosts()"
                 << pyre::journal::endl;
        incrcosts(row,col).negcost=LARGESHORT;
      }else{
        incrcosts(row,col).negcost=templong;
      }
      if(incrcosts(row,col).negcost>maxcost){
        maxcost=incrcosts(row,col).negcost;
      }
    }
  }

  /* loop over column arcs and convolve */
  for(row=nrow-1;row<2*nrow-1;row++){
    for(col=0;col<ncol-1;col++){
      templong=2*incrcosts(row,col).poscost;
      n=2.0;
      if(row!=nrow-1){
        templong+=incrcosts(row-1,col).poscost;
        n+=1.0;
      }
      if(row!=2*nrow-2){
        templong+=incrcosts(row+1,col).poscost;
        n+=1.0;
      }
      templong=LRound(templong/n);
      if(templong>LARGESHORT){
        fflush(NULL);
        warnings << pyre::journal::at(__HERE__)
                 << "WARNING: COSTS CLIPPED IN ThickenCosts()"
                 << pyre::journal::endl;
        incrcosts(row,col).negcost=LARGESHORT;
      }else{
        incrcosts(row,col).negcost=templong;
      }
      if(incrcosts(row,col).negcost>maxcost){
        maxcost=incrcosts(row,col).negcost;
      }
    }
  }

  /* return maximum cost */
  return(maxcost);

}


/* function: RegionsNeighborNode()
 * -------------------------------
 * Return the neighboring node of the given node corresponding to the
 * given arc number.
 */
static
nodeT *RegionsNeighborNode(nodeT *node1, long *arcnumptr, Array2D<nodeT>& nodes,
                           long *arcrowptr, long *arccolptr,
                           long nrow, long ncol){

  long row, col;

  row=node1->row;
  col=node1->col;

  while(TRUE){
    switch((*arcnumptr)++){
    case 0:
      if(col!=ncol-1){
        *arcrowptr=nrow-1+row;
        *arccolptr=col;
        return(&nodes(row,col+1));
      }
      break;
    case 1:
      if(row!=nrow-1){
      *arcrowptr=row;
      *arccolptr=col;
        return(&nodes(row+1,col));
      }
      break;
    case 2:
      if(col!=0){
        *arcrowptr=nrow-1+row;
        *arccolptr=col-1;
        return(&nodes(row,col-1));
      }
      break;
    case 3:
      if(row!=0){
        *arcrowptr=row-1;
        *arccolptr=col;
        return(&nodes(row-1,col));
      }
      break;
    default:
      return(NULL);
    }
  }
}


/* function: ClearBuckets()
 * ------------------------
 * Removes any nodes in the bucket data structure passed, and resets
 * their distances to VERYFAR.  Assumes bukets indexed from 0.
 */
static
int ClearBuckets(bucketT *bkts){

  nodeT *currentnode, *nextnode;
  long i;

  /* loop over all buckets */
  for(i=0;i<bkts->size;i++){

    /* clear the bucket */
    nextnode=bkts->bucketbase[i];
    while(nextnode!=NULL){
      currentnode=nextnode;
      nextnode=currentnode->next;
      currentnode->group=NOTINBUCKET;
      currentnode->outcost=VERYFAR;
      currentnode->pred=NULL;
    }
    bkts->bucketbase[i]=NULL;
  }

  /* reset bucket parameters */
  bkts->minind=0;
  bkts->maxind=bkts->size-1;
  bkts->wrapped=FALSE;

  /* done */
  return(0);
}


/* function: MergeRegions()
 * ------------------------
 *
 */
static
int MergeRegions(Array2D<nodeT>& nodes, nodeT *source, Array1D<long>& regionsizes,
                 long closestregion, long nrow, long ncol){

  long nextnodelistlen, nextnodelistnext, arcnum, arcrow, arccol, regionnum;
  nodeT *from, *to;

  /* initialize */
  nextnodelistlen=INITARRSIZE;
  auto nextnodelist=Array1D<nodeT*>(nextnodelistlen);
  nextnodelist[0]=source;
  nextnodelistnext=1;
  regionnum=source->incost;

  /* find all nodes in current region and switch their regions */
  while(nextnodelistnext){
    from=nextnodelist[--nextnodelistnext];
    from->incost=closestregion;
    arcnum=0;
    while((to=RegionsNeighborNode(from,&arcnum,nodes,
                                  &arcrow,&arccol,nrow,ncol))!=NULL){
      if(to->incost==regionnum){
        if(nextnodelistnext>=nextnodelistlen){
          nextnodelistlen+=INITARRSIZE;
          nextnodelist.conservativeResize(nextnodelistlen);
        }
        nextnodelist[nextnodelistnext++]=to;
      }
    }
  }

  /* update size of region to which we are merging */
  regionsizes[closestregion]+=regionsizes[regionnum];

  return(0);

}


/* function: RenumberRegion()
 * --------------------------
 *
 */
static
int RenumberRegion(Array2D<nodeT>& nodes, nodeT *source, long newnum,
                   long nrow, long ncol){

  long nextnodelistlen, nextnodelistnext, arcnum, arcrow, arccol, regionnum;
  nodeT *from, *to;

  /* initialize */
  nextnodelistlen=INITARRSIZE;
  auto nextnodelist=Array1D<nodeT*>(nextnodelistlen);
  nextnodelist[0]=source;
  nextnodelistnext=1;
  regionnum=source->incost;

  /* find all nodes in current region and switch their regions */
  while(nextnodelistnext){
    from=nextnodelist[--nextnodelistnext];
    from->incost=newnum;
    arcnum=0;
    while((to=RegionsNeighborNode(from,&arcnum,nodes,
                                  &arcrow,&arccol,nrow,ncol))!=NULL){
      if(to->incost==regionnum){
        if(nextnodelistnext>=nextnodelistlen){
          nextnodelistlen+=INITARRSIZE;
          nextnodelist.conservativeResize(nextnodelistlen);
        }
        nextnodelist[nextnodelistnext++]=to;
      }
    }
  }

  return(0);

}


/* function: AssembleTiles()
 * -------------------------
 */
template<class CostTag>
int AssembleTiles(outfileT *outfiles, paramT *params, 
                  long nlines, long linelen, CostTag tag){

  long tilerow, tilecol, ntilerow, ntilecol, ntiles, rowovrlp, colovrlp;
  long i, j, k, ni, nj, dummylong;
  long nrow, ncol, prevnrow, prevncol, nextnrow, nextncol;
  long n, ncycle, nflowdone, nflow, candidatelistsize, candidatebagsize;
  long nnodes, maxnflowcycles, arclen, narcs, sourcetilenum, flowmax;
  long nincreasedcostiter;
  double avgarclen;
  totalcostT totalcost, oldtotalcost, mintotalcost;
  nodeT *source;
  signed char notfirstloop;
  bucketT bkts[1]={};

  Array1D<candidateT> candidatebag, candidatelist;

  /* set up */
  auto info=pyre::journal::info_t("isce3.unwrap.snaphu");
  info << pyre::journal::at(__HERE__)
       << "Assembling tiles"
       << pyre::journal::endl;
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  ntiles=ntilerow*ntilecol;
  rowovrlp=params->rowovrlp;
  colovrlp=params->colovrlp;
  ni=ceil((nlines+(ntilerow-1)*rowovrlp)/(double )ntilerow);
  nj=ceil((linelen+(ntilecol-1)*colovrlp)/(double )ntilecol);
  nrow=0;
  ncol=0;
  flowmax=params->scndryarcflowmax;
  prevnrow=0;

  /* get memory */
  auto regions=Array2D<short>(ni,nj);
  auto nextregions=Array2D<short>(ni,nj);
  auto lastregions=Array2D<short>(ni,nj);
  auto regionsabove=Array2D<short>(1,nj);
  auto regionsbelow=Array2D<short>(1,nj);
  auto unwphase=Array2D<float>(ni,nj);
  auto nextunwphase=Array2D<float>(ni,nj);
  auto lastunwphase=Array2D<float>(ni,nj);
  auto unwphaseabove=Array2D<float>(1,nj);
  auto unwphasebelow=Array2D<float>(1,nj);

  auto scndrynodes=Array2D<nodeT>(ntiles,0);
  auto nodesupp=Array2D<nodesuppT>(ntiles,0);
  auto scndryarcs=Array2D<scndryarcT>(ntiles,0);
  auto scndrycosts=Array2D<Array1D<long>>(ntiles,0);
  auto nscndrynodes=Array1D<int>(ntiles);
  auto nscndryarcs=Array1D<int>(ntiles);
  auto totarclens=Array1D<long>(ntiles);
  auto bulkoffsets=Array2D<short>(ntilerow,ntilecol);

  using Cost=typename CostTag::Cost;
  auto costs=MakeRowColArray2D<Cost>(ni+2,nj+2);
  auto nextcosts=MakeRowColArray2D<Cost>(ni+2,nj+2);
  auto lastcosts=MakeRowColArray2D<Cost>(ni+2,nj+2);
  auto costsabove=Array2D<Cost>(1,nj);
  auto costsbelow=Array2D<Cost>(1,nj);

  auto regionsptr=&regions;
  auto nextregionsptr=&nextregions;
  auto lastregionsptr=&lastregions;
  auto unwphaseptr=&unwphase;
  auto nextunwphaseptr=&nextunwphase;
  auto lastunwphaseptr=&lastunwphase;
  auto costsptr=&costs;
  auto nextcostsptr=&nextcosts;
  auto lastcostsptr=&lastcosts;

  Array2D<short>* tempregionsptr=nullptr;
  Array2D<float>* tempunwphaseptr=nullptr;
  Array2D<Cost>* tempcostsptr=nullptr;

  /* trace regions and parse secondary nodes and arcs for each tile */
  bulkoffsets(0,0)=0;
  for(tilerow=0;tilerow<ntilerow;tilerow++){
    for(tilecol=0;tilecol<ntilecol;tilecol++){

      /* read region, unwrapped phase, and flow data */
      if(tilecol==0){
        ReadNextRegion(tilerow,0,nlines,linelen,outfiles,params,
                       nextregionsptr,nextunwphaseptr,nextcostsptr,
                       &nextnrow,&nextncol);
        prevnrow=nrow;
        nrow=nextnrow;
      }
      prevncol=ncol;
      ncol=nextncol;
      tempregionsptr=lastregionsptr;
      lastregionsptr=regionsptr;
      regionsptr=nextregionsptr;
      nextregionsptr=tempregionsptr;
      tempunwphaseptr=lastunwphaseptr;
      lastunwphaseptr=unwphaseptr;
      unwphaseptr=nextunwphaseptr;
      nextunwphaseptr=tempunwphaseptr;
      tempcostsptr=lastcostsptr;
      lastcostsptr=costsptr;
      costsptr=nextcostsptr;
      nextcostsptr=tempcostsptr;
      if(tilecol!=ntilecol-1){
        ReadNextRegion(tilerow,tilecol+1,nlines,linelen,outfiles,params,
                       nextregionsptr,nextunwphaseptr,nextcostsptr,
                       &nextnrow,&nextncol);
      }
      ReadEdgesAboveAndBelow(tilerow,tilecol,nlines,linelen,params,
                             outfiles,regionsabove,regionsbelow,
                             unwphaseabove,unwphasebelow,
                             costsabove,costsbelow);

      /* trace region edges to form nodes and arcs */
      TraceRegions(*regionsptr,*nextregionsptr,*lastregionsptr,regionsabove,regionsbelow,
                   *unwphaseptr,*nextunwphaseptr,*lastunwphaseptr,unwphaseabove,
                   unwphasebelow,*costsptr,*nextcostsptr,*lastcostsptr,costsabove,
                   costsbelow,prevnrow,prevncol,tilerow,tilecol,
                   nrow,ncol,scndrynodes,nodesupp,scndryarcs,
                   scndrycosts,nscndrynodes,nscndryarcs,totarclens,
                   bulkoffsets,params,tag);

    }
  }

  /* scale costs based on average number of primary arcs per secondary arc */
  arclen=0;
  narcs=0;
  for(i=0;i<ntiles;i++){
    arclen+=totarclens[i];
    narcs+=nscndryarcs[i];
  }
  avgarclen=arclen/narcs;

  /* may need to adjust scaling so fewer costs clipped */
  for(i=0;i<ntiles;i++){
    for(j=0;j<nscndryarcs[i];j++){
      if(scndrycosts(i,j)[2*flowmax+1]!=ZEROCOSTARC){
        for(k=1;k<=2*flowmax;k++){
          scndrycosts(i,j)[k]=(long )ceil(scndrycosts(i,j)[k]/avgarclen);
        }
        scndrycosts(i,j)[2*flowmax+1]=LRound(scndrycosts(i,j)[2*flowmax+1]
                                              /avgarclen);
        if(scndrycosts(i,j)[2*flowmax+1]<0){
          scndrycosts(i,j)[2*flowmax+1]=0;
        }
      }
    }
  }

  /* get memory for nongrid arrays of secondary network problem */
  auto maxnscndryarcs=*std::max_element(nscndryarcs.data(),nscndryarcs.data()+ntiles);
  nnodes=0;
  for(i=0;i<ntiles;i++){
    nnodes+=nscndrynodes[i];
  }

  Array2D<short> scndryflows=Array2D<short>::Zero(ntiles,maxnscndryarcs);
  Array2D<signed char> iscandidate=Array2D<signed char>::Zero(ntiles,maxnscndryarcs);

  auto scndryapexes=Array2D<nodeT*>(ntiles,maxnscndryarcs);
  auto incrcosts=Array2D<incrcostT>(ntiles,maxnscndryarcs);

  /* set up network for secondary solver */
  InitNetwork(scndryflows,&dummylong,&ncycle,&nflowdone,&dummylong,&nflow,
              &candidatebagsize,&candidatebag,&candidatelistsize,
              &candidatelist,NULL,NULL,bkts,&dummylong,NULL,NULL,NULL,
              NULL,NULL,NULL,NULL,ntiles,0,&notfirstloop,&totalcost,params);
  oldtotalcost=totalcost;
  mintotalcost=totalcost;
  nincreasedcostiter=0;

  /* set pointers to functions for nongrid secondary network */
  NonGridCostTag nongridcosttag;
  SetNonGridNetworkFunctionPointers();

  /* solve the secondary network problem */
  /* main loop: loop over flow increments and sources */
  info << pyre::journal::at(__HERE__)
       << "Running optimizer for secondary network"
       << pyre::journal::endl
       << "Number of nodes in secondary network: " << nnodes
       << pyre::journal::endl;
  maxnflowcycles=LRound(nnodes*params->maxcyclefraction);
  while(TRUE){ 
 
    info << pyre::journal::at(__HERE__)
         << "Flow increment: " << nflow << "  (Total improvements: "
         << ncycle << ")"
         << pyre::journal::endl;

    /* set up the incremental (residual) cost arrays */
    SetupIncrFlowCosts(scndrycosts,incrcosts,scndryflows,nflow,ntiles,
                       ntiles,nscndryarcs,params,nongridcosttag);

    /* set the tree root (equivalent to source of shortest path problem) */
    sourcetilenum=(long )ntilecol*floor(ntilerow/2.0)+floor(ntilecol/2.0);
    source=&scndrynodes(sourcetilenum,0);

    /* set up network variables for tree solver */
    SetupTreeSolveNetwork(scndrynodes,NULL,scndryapexes,iscandidate,
                          ntiles,nscndrynodes,ntiles,nscndryarcs,
                          ntiles,0);

    /* run the solver, and increment nflowdone if no cycles are found */
    Array2D<float> dummy;
    n=TreeSolve(scndrynodes,nodesupp,NULL,source,&candidatelist,&candidatebag,
                &candidatelistsize,&candidatebagsize,bkts,scndryflows,
                scndrycosts,incrcosts,scndryapexes,iscandidate,0,
                nflow,dummy,dummy,NULL,ntiles,nscndrynodes,ntiles,nscndryarcs,
                ntiles,0,NULL,nnodes,params,nongridcosttag);

    /* evaluate and save the total cost (skip if first loop through nflow) */
    if(notfirstloop){
      oldtotalcost=totalcost;
      totalcost=EvaluateTotalCost(scndrycosts,scndryflows,ntiles,0,
                                  nscndryarcs,params,nongridcosttag);
      if(totalcost<mintotalcost){
        mintotalcost=totalcost;
      }
      if(totalcost>oldtotalcost || (n>0 && totalcost==oldtotalcost)){
        fflush(NULL);
        info << pyre::journal::at(__HERE__)
             << "Caution: Unexpected increase in total cost"
             << pyre::journal::endl;
      }
      if(totalcost>mintotalcost){
        nincreasedcostiter++;
      }else{
        nincreasedcostiter=0;
      }
    }

    /* consider this flow increment done if not too many neg cycles found */
    ncycle+=n;
    if(n<=maxnflowcycles){
      nflowdone++;
    }else{
      nflowdone=1;
    }

    /* break if total cost increase is sustained */
    if(nincreasedcostiter>=params->maxflow){
      fflush(NULL);
      auto warnings=pyre::journal::warning_t("isce3.unwrap.snaphu");
      warnings << pyre::journal::at(__HERE__)
               << "WARNING: Unexpected sustained increase in total cost."
               << "  Breaking loop"
               << pyre::journal::endl;
      break;
    }


    /* break if we're done with all flow increments or problem is convex */
    if(nflowdone>=params->maxflow){
      break;
    }

    /* update flow increment */
    nflow++;
    if(nflow>params->maxflow){
      nflow=1;
      notfirstloop=TRUE;
    }

  } /* end loop until no more neg cycles */

  /* assemble connected component files if needed */
  if(strlen(outfiles->conncompfile)){
    AssembleTileConnComps(linelen,nlines,outfiles,params);
  }

  /* integrate phase from secondary network problem */
  IntegrateSecondaryFlows(linelen,nlines,scndrynodes,nodesupp,scndryarcs,
                          nscndryarcs,scndryflows,bulkoffsets,outfiles,params);

  /* remove temporary tile log files and tile directory */
  if(params->rmtmptile){
    fflush(NULL);
    info << pyre::journal::at(__HERE__)
         << "Removing temporary directory " << params->tiledir
         << pyre::journal::endl;
    for(tilerow=0;tilerow<ntilerow;tilerow++){
      for(tilecol=0;tilecol<ntilecol;tilecol++){
        auto filename=std::string(params->tiledir)+"/"
          +LOGFILEROOT+std::to_string(tilerow)+"_"+std::to_string(tilecol);
        unlink(filename.c_str());
      }
    }
    rmdir(params->tiledir);
  }

  /* Give notice about increasing overlap if there are edge artifacts */
  if(params->rowovrlp<ni || params->colovrlp<nj){
    fflush(NULL);
    info << pyre::journal::at(__HERE__)
         << "SUGGESTION: Try increasing tile overlap and/or size"
         << " if solution has edge artifacts"
         << pyre::journal::endl;
  }

  /* done */
  return(0);
}


/* function: ReadNextRegion()
 * --------------------------
 */
template<class Cost>
static
int ReadNextRegion(long tilerow, long tilecol, long nlines, long linelen,
                   outfileT *outfiles, paramT *params,
                   Array2D<short>* nextregionsptr, Array2D<float>* nextunwphaseptr,
                   Array2D<Cost>* nextcostsptr,
                   long *nextnrowptr, long *nextncolptr){

  long nexttilelinelen, nexttilenlines, costtypesize=0;
  tileparamT nexttileparams[1]={};
  outfileT nexttileoutfiles[1]={};
  char nextfile[MAXSTRLEN]={};
  char path[MAXSTRLEN]={}, basename[MAXSTRLEN]={};
  
  /* size of the data type for holding cost data depends on cost mode */
  if(params->p<0){
    if(params->costmode==TOPO || params->costmode==DEFO){
      costtypesize=sizeof(costT);
    }else if(params->costmode==SMOOTH){
      costtypesize=sizeof(smoothcostT);
    }
  }else if(params->bidirlpn){
    costtypesize=sizeof(bidircostT);
  }else{
    costtypesize=sizeof(short);
  }

  /* use SetupTile() to set filenames only; tile params overwritten below */
  SetupTile(nlines,linelen,params,nexttileparams,outfiles,nexttileoutfiles,
            tilerow,tilecol);
  nexttilenlines=nexttileparams->nrow;
  nexttilelinelen=nexttileparams->ncol;

  /* set tile parameters, overwriting values set by SetupTile() above */
  SetTileReadParams(nexttileparams,nexttilenlines,nexttilelinelen,
                    tilerow,tilecol,nlines,linelen,params);

  /* read region data */
  ParseFilename(outfiles->outfile,path,basename);
  auto tempstring=std::string(params->tiledir)+"/"+TMPTILEROOT+basename+"_"
    +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
      +std::to_string(nexttilelinelen)+REGIONSUFFIX;
  StrNCopy(nextfile,tempstring.c_str(),MAXSTRLEN);
  Read2DArray(nextregionsptr,nextfile,
              nexttilelinelen,nexttilenlines,
              nexttileparams,sizeof(short *),sizeof(short));

  /* read unwrapped phase data */
  if(TMPTILEOUTFORMAT==ALT_LINE_DATA){
    ReadAltLineFilePhase(nextunwphaseptr,nexttileoutfiles->outfile,
                         nexttilelinelen,nexttilenlines,nexttileparams);
  }else if(TMPTILEOUTFORMAT==FLOAT_DATA){
    Read2DArray(nextunwphaseptr,nexttileoutfiles->outfile,
                nexttilelinelen,nexttilenlines,nexttileparams,
                sizeof(float *),sizeof(float));
  }else{
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Cannot read format of unwrapped phase tile data");
  }

  /* read cost data */
  Read2DRowColFile(nextcostsptr,nexttileoutfiles->costoutfile,
                   nexttilelinelen,nexttilenlines,nexttileparams,
                   costtypesize);

  /* flip sign of wrapped phase if flip flag is set */
  FlipPhaseArraySign(*nextunwphaseptr,params,
                     nexttileparams->nrow,nexttileparams->ncol);

  /* set outputs */
  (*nextnrowptr)=nexttileparams->nrow;
  (*nextncolptr)=nexttileparams->ncol;
  return(0);

}

/* function: SetTileReadParams()
 * -----------------------------
 * Set parameters for reading the nonoverlapping piece of each tile.  
 * ni and nj are the numbers of rows and columns in this particular tile.
 * The meanings of these variables are different for the last row 
 * and column.
 */
static
int SetTileReadParams(tileparamT *tileparams, long nexttilenlines, 
                      long nexttilelinelen, long tilerow, long tilecol, 
                      long /*nlines*/, long /*linelen*/, paramT *params){

  long rowovrlp, colovrlp;

  /* set temporary variables */
  rowovrlp=params->rowovrlp;
  colovrlp=params->colovrlp;

  /* row parameters */
  if(tilerow==0){
    tileparams->firstrow=0;
  }else{
    tileparams->firstrow=ceil(rowovrlp/2.0);
  }
  if(tilerow!=params->ntilerow-1){
    tileparams->nrow=nexttilenlines-floor(rowovrlp/2.0)-tileparams->firstrow;
  }else{
    tileparams->nrow=nexttilenlines-tileparams->firstrow;
  }

  /* column parameters */
  if(tilecol==0){
    tileparams->firstcol=0;
  }else{
    tileparams->firstcol=ceil(colovrlp/2.0);
  }
  if(tilecol!=params->ntilecol-1){
    tileparams->ncol=nexttilelinelen-floor(colovrlp/2.0)-tileparams->firstcol;
  }else{
    tileparams->ncol=nexttilelinelen-tileparams->firstcol;
  }
  return(0);
}


/* function: ReadEdgesAboveAndBelow()
 * ----------------------------------
 */
template<class Cost>
static
int ReadEdgesAboveAndBelow(long tilerow, long tilecol, long nlines,
                           long linelen, paramT *params, outfileT *outfiles,
                           Array2D<short>& regionsabove, Array2D<short>& regionsbelow,
                           Array2D<float>& unwphaseabove, Array2D<float>& unwphasebelow,
                           Array2D<Cost>& costsabove, Array2D<Cost>& costsbelow){

  long ni, nj, readtilelinelen, readtilenlines, costtypesize=0;
  long ntilerow, ntilecol, rowovrlp, colovrlp;
  tileparamT tileparams[1]={};
  outfileT outfilesabove[1]={}, outfilesbelow[1]={};
  char readregionfile[MAXSTRLEN]={};
  char path[MAXSTRLEN]={}, basename[MAXSTRLEN]={};
  std::string tempstring;
  
  /* set temporary variables */
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  rowovrlp=params->rowovrlp;
  colovrlp=params->colovrlp;
  ni=ceil((nlines+(ntilerow-1)*rowovrlp)/(double )ntilerow);
  nj=ceil((linelen+(ntilecol-1)*colovrlp)/(double )ntilecol);

  /* size of the data type for holding cost data depends on cost mode */
  if(params->p<0){
    if(params->costmode==TOPO || params->costmode==DEFO){
      costtypesize=sizeof(costT);
    }else if(params->costmode==SMOOTH){
      costtypesize=sizeof(smoothcostT);
    }
  }else if(params->bidirlpn){
    costtypesize=sizeof(bidircostT);
  }else{
    costtypesize=sizeof(short);
  }

  /* set names of files with SetupTile() */
  /* tile parameters set by SetupTile() will be overwritten below */
  if(tilerow!=0){
    SetupTile(nlines,linelen,params,tileparams,outfiles,outfilesabove,
              tilerow-1,tilecol);
  }
  if(tilerow!=ntilerow-1){
    SetupTile(nlines,linelen,params,tileparams,outfiles,outfilesbelow,
              tilerow+1,tilecol);
  }

  /* set some reading parameters */
  if(tilecol==0){
    tileparams->firstcol=0;
  }else{
    tileparams->firstcol=ceil(colovrlp/2.0);
  }
  if(tilecol!=params->ntilecol-1){
    readtilelinelen=nj;
    tileparams->ncol=readtilelinelen-floor(colovrlp/2.0)-tileparams->firstcol;
  }else{
    readtilelinelen=linelen-(ntilecol-1)*(nj-colovrlp);
    tileparams->ncol=readtilelinelen-tileparams->firstcol;
  }
  tileparams->nrow=1;

  /* read last line of tile above */
  readtilenlines=ni;
  if(tilerow!=0){
    tileparams->firstrow=readtilenlines-floor(rowovrlp/2.0)-1;

    /* read region data */
    ParseFilename(outfiles->outfile,path,basename);
    tempstring=std::string(params->tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow-1)+"_"+std::to_string(tilecol)+"."
        +std::to_string(readtilelinelen)+REGIONSUFFIX;
    StrNCopy(readregionfile,tempstring.c_str(),MAXSTRLEN);
    Read2DArray(&regionsabove,readregionfile,
                readtilelinelen,readtilenlines,
                tileparams,sizeof(short *),sizeof(short));

    /* read unwrapped phase data */
    if(TMPTILEOUTFORMAT==ALT_LINE_DATA){
      ReadAltLineFilePhase(&unwphaseabove,outfilesabove->outfile,
                           readtilelinelen,readtilenlines,tileparams);
    }else if(TMPTILEOUTFORMAT==FLOAT_DATA){
      Read2DArray(&unwphaseabove,outfilesabove->outfile,
                  readtilelinelen,readtilenlines,tileparams,
                  sizeof(float *),sizeof(float));
    }

    /* flip sign of wrapped phase if flip flag is set */
    FlipPhaseArraySign(unwphaseabove,params,
                       tileparams->nrow,tileparams->ncol);

    /* read costs data */
    tileparams->firstrow--;
    Read2DRowColFileRows(&costsabove,outfilesabove->costoutfile,
                         readtilelinelen,readtilenlines,tileparams,
                         costtypesize);

    /* remove temporary tile cost file unless told to save it */
    if(params->rmtmptile && !strlen(outfiles->costoutfile)){
      unlink(outfilesabove->costoutfile);
    }
  }

  /* read first line of tile below */
  if(tilerow!=ntilerow-1){
    if(tilerow==params->ntilerow-2){
      readtilenlines=nlines-(ntilerow-1)*(ni-rowovrlp);
    }
    tileparams->firstrow=ceil(rowovrlp/2.0);

    /* read region data */
    ParseFilename(outfiles->outfile,path,basename);
    tempstring=std::string(params->tiledir)+"/"+TMPTILEROOT+basename+"_"
      +std::to_string(tilerow+1)+"_"+std::to_string(tilecol)+"."
        +std::to_string(readtilelinelen)+REGIONSUFFIX;
    StrNCopy(readregionfile,tempstring.c_str(),MAXSTRLEN);
    Read2DArray(&regionsbelow,readregionfile,
                readtilelinelen,readtilenlines,
                tileparams,sizeof(short *),sizeof(short));

    /* read unwrapped phase data */
    if(TMPTILEOUTFORMAT==ALT_LINE_DATA){
      ReadAltLineFilePhase(&unwphasebelow,outfilesbelow->outfile,
                           readtilelinelen,readtilenlines,tileparams);
    }else if(TMPTILEOUTFORMAT==FLOAT_DATA){
      Read2DArray(&unwphasebelow,outfilesbelow->outfile,
                  readtilelinelen,readtilenlines,tileparams,
                  sizeof(float *),sizeof(float));
    }

    /* flip the sign of the wrapped phase if flip flag is set */
    FlipPhaseArraySign(unwphasebelow,params,
                       tileparams->nrow,tileparams->ncol);

    /* read costs data */
    Read2DRowColFileRows(&costsbelow,outfilesbelow->costoutfile,
                         readtilelinelen,readtilenlines,tileparams,
                         costtypesize);

  }else{

    /* remove temporoary tile cost file for last row unless told to save it */
    if(params->rmtmptile && !strlen(outfiles->costoutfile)){
      SetupTile(nlines,linelen,params,tileparams,outfiles,outfilesbelow,
                tilerow,tilecol);
      unlink(outfilesbelow->costoutfile);
    }
  }

  /* done */
  return(0);
  
}


/* function: TraceRegions()
 * ------------------------
 * Trace edges of region data to form nodes and arcs of secondary
 * (ie, region-level) network problem.  Primary nodes and arcs are
 * those of the original, pixel-level network problem.  Flows along
 * edges are computed knowing the unwrapped phase values of edges
 * of adjacent tiles.  Costs along edges are approximated in that they
 * are calculated from combining adjacent cost parameters, not from 
 * using the exact method in BuildCostArrays().
 */
template<class CostTag>
static
int TraceRegions(Array2D<short>& regions, Array2D<short>& nextregions, Array2D<short>& lastregions,
                 Array2D<short>& regionsabove, Array2D<short>& regionsbelow, Array2D<float>& unwphase,
                 Array2D<float>& nextunwphase, Array2D<float>& lastunwphase,
                 Array2D<float>& unwphaseabove, Array2D<float>& unwphasebelow,
                 Array2D<typename CostTag::Cost>& costs, Array2D<typename CostTag::Cost>& nextcosts,
                 Array2D<typename CostTag::Cost>& lastcosts, Array2D<typename CostTag::Cost>& costsabove,
                 Array2D<typename CostTag::Cost>& costsbelow, long prevnrow, long prevncol, long tilerow,
                 long tilecol, long nrow, long ncol, Array2D<nodeT>& scndrynodes,
                 Array2D<nodesuppT>& nodesupp, Array2D<scndryarcT>& scndryarcs,
                 Array2D<Array1D<long>>& scndrycosts, Array1D<int>& nscndrynodes,
                 Array1D<int>& nscndryarcs, Array1D<long>& totarclens, Array2D<short>& bulkoffsets,
                 paramT *params, CostTag tag){

  long i, j, row, col, nnrow, nncol, tilenum;
  long nnewnodes, nnewarcs, npathsout, flowmax, totarclen;
  long nupdatednontilenodes, updatednontilenodesize, ntilecol;
  nodeT *from, *to, *nextnode, *tempnode;
  nodesuppT *fromsupp, *tosupp;

  /* initialize */
  ntilecol=params->ntilecol;
  nnrow=nrow+1;
  nncol=ncol+1;
  auto primarynodes=Array2D<nodeT>(nnrow,nncol);
  for(row=0;row<nnrow;row++){
    for(col=0;col<nncol;col++){
      primarynodes(row,col).row=row;
      primarynodes(row,col).col=col;
      primarynodes(row,col).group=NOTINBUCKET;
      primarynodes(row,col).pred=NULL;
      primarynodes(row,col).next=NULL;
    }
  }
  nextnode=&primarynodes(0,0);
  tilenum=tilerow*ntilecol+tilecol;
  nnewnodes=0;
  nnewarcs=0;
  totarclen=0;
  flowmax=params->scndryarcflowmax;
  updatednontilenodesize=INITARRSIZE;
  nupdatednontilenodes=0;

  /* get memory */
  auto updatednontilenodes=Array1D<nodeT*>(updatednontilenodesize);
  auto inontilenodeoutarc=Array1D<short>(updatednontilenodesize);
  auto flows=MakeRowColArray2D<short>(nrow+1,ncol+1);
  auto rightedgeflows=Array2D<short>(nrow,1);
  auto leftedgeflows=Array2D<short>(nrow,1);
  auto upperedgeflows=Array2D<short>(1,ncol);
  auto loweredgeflows=Array2D<short>(1,ncol);

  using Cost=typename CostTag::Cost;
  auto rightedgecosts=Array2D<Cost>(nrow,1);
  auto leftedgecosts=Array2D<Cost>(nrow,1);
  auto upperedgecosts=Array2D<Cost>(1,ncol);
  auto loweredgecosts=Array2D<Cost>(1,ncol);

  /* parse flows for this tile */
  CalcFlow(unwphase,&flows,nrow,ncol);

  /* set up cost and flow arrays for boundaries */
  SetUpperEdge(ncol,tilerow,tilecol,costs,costsabove,unwphase,unwphaseabove,
               upperedgecosts,upperedgeflows,params, bulkoffsets);
  SetLowerEdge(nrow,ncol,tilerow,tilecol,costs,costsbelow,unwphase,
               unwphasebelow,loweredgecosts,loweredgeflows,
               params,bulkoffsets);
  SetLeftEdge(nrow,prevncol,tilerow,tilecol,costs,lastcosts,unwphase,
              lastunwphase,leftedgecosts,leftedgeflows,params, bulkoffsets);
  SetRightEdge(nrow,ncol,tilerow,tilecol,costs,nextcosts,unwphase, 
               nextunwphase,rightedgecosts,rightedgeflows,
               params,bulkoffsets);

  /* trace edges between regions */
  while(nextnode!=NULL){

    /* get next primary node from stack */
    from=nextnode;
    nextnode=nextnode->next;
    from->group=NOTINBUCKET;

    /* find number of paths out of from node */
    npathsout=FindNumPathsOut(from,params,tilerow,tilecol,nnrow,nncol,regions,
                              nextregions,lastregions,regionsabove,
                              regionsbelow,prevncol);

    /* secondary node exists if region edges fork */
    if(npathsout>2){

      /* mark primary node to indicate that secondary node exists for it */
      from->group=ONTREE;

      /* create secondary node if not already created in another tile */
      if((from->row!=0 || tilerow==0) && (from->col!=0 || tilecol==0)){

        /* create the secondary node */
        nnewnodes++;
        if(nnewnodes > SHRT_MAX){
          fflush(NULL);
          throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                  "Exceeded maximum number of secondary nodes. Decrease "
                  "TILECOSTTHRESH and/or increase MINREGIONSIZE");
        }

        if(nnewnodes>scndrynodes.cols()){
          auto nnewcols=std::max(nnewnodes,2*scndrynodes.cols());
          scndrynodes.conservativeResize(Eigen::NoChange,nnewcols);
        }
        if(nnewnodes>nodesupp.cols()){
          auto nnewcols=std::max(nnewnodes,2*nodesupp.cols());
          nodesupp.conservativeResize(Eigen::NoChange,nnewcols);
        }
        scndrynodes(tilenum,nnewnodes-1).row=tilenum;
        scndrynodes(tilenum,nnewnodes-1).col=nnewnodes-1;
        nodesupp(tilenum,nnewnodes-1).row=from->row;
        nodesupp(tilenum,nnewnodes-1).col=from->col;
        nodesupp(tilenum,nnewnodes-1).noutarcs=0;
        nodesupp(tilenum,nnewnodes-1).neighbornodes={};
        nodesupp(tilenum,nnewnodes-1).outarcs={};
      }

      /* create the secondary arc to this node if it doesn't already exist */
      if(from->pred!=NULL
         && ((from->row==from->pred->row && (from->row!=0 || tilerow==0))
             || (from->col==from->pred->col && (from->col!=0 || tilecol==0)))){

        TraceSecondaryArc(from,scndrynodes,nodesupp,scndryarcs,scndrycosts,
                          &nnewnodes,&nnewarcs,tilerow,tilecol,flowmax,
                          nrow,ncol,prevnrow,prevncol,params,costs,
                          rightedgecosts,loweredgecosts,leftedgecosts,
                          upperedgecosts,flows,rightedgeflows,loweredgeflows, 
                          leftedgeflows,upperedgeflows,&updatednontilenodes,
                          &nupdatednontilenodes,&updatednontilenodesize,
                          &inontilenodeoutarc,&totarclen,tag);
      }
    }

    /* scan neighboring primary nodes and place path candidates into stack */
    RegionTraceCheckNeighbors(from,&nextnode,primarynodes,regions,
                              nextregions,lastregions,regionsabove,
                              regionsbelow,tilerow,tilecol,nnrow,nncol,
                              scndrynodes,nodesupp,scndryarcs,&nnewnodes,
                              &nnewarcs,flowmax,nrow,ncol,prevnrow,prevncol,
                              params,costs,rightedgecosts,loweredgecosts,
                              leftedgecosts,upperedgecosts,flows,
                              rightedgeflows,loweredgeflows,leftedgeflows,
                              upperedgeflows,scndrycosts,&updatednontilenodes,
                              &nupdatednontilenodes,&updatednontilenodesize,
                              &inontilenodeoutarc,&totarclen,tag);
  }

  /* reset temporary secondary node and arc pointers in data structures */
  /* secondary node row, col stored level, incost of primary node pointed to */

  /* update nodes in this tile */
  for(i=0;i<nnewnodes;i++){
    for(j=0;j<nodesupp(tilenum,i).noutarcs;j++){
      tempnode=nodesupp(tilenum,i).neighbornodes[j];
      nodesupp(tilenum,i).neighbornodes[j]
        =&scndrynodes(tempnode->level,tempnode->incost);
    }
  }

  /* update nodes not in this tile that were affected (that have new arcs) */
  for(i=0;i<nupdatednontilenodes;i++){
    row=updatednontilenodes[i]->row;
    col=updatednontilenodes[i]->col;
    j=inontilenodeoutarc[i];
    tempnode=nodesupp(row,col).neighbornodes[j];
    nodesupp(row,col).neighbornodes[j]
      =&scndrynodes(tempnode->level,tempnode->incost);
  }

  /* update secondary arcs */
  for(i=0;i<nnewarcs;i++){

    /* update node pointers in secondary arc structure */
    tempnode=scndryarcs(tilenum,i).from;
    scndryarcs(tilenum,i).from
      =&scndrynodes(tempnode->level,tempnode->incost);
    from=scndryarcs(tilenum,i).from;
    tempnode=scndryarcs(tilenum,i).to;
    scndryarcs(tilenum,i).to
      =&scndrynodes(tempnode->level,tempnode->incost);
    to=scndryarcs(tilenum,i).to;

    /* update secondary arc pointers in nodesupp strcutres */
    fromsupp=&nodesupp(from->row,from->col);
    j=0;
    while(fromsupp->neighbornodes[j]!=to){
      j++;
    }
    fromsupp->outarcs[j]=&scndryarcs(tilenum,i);
    tosupp=&nodesupp(to->row,to->col);
    j=0;
    while(tosupp->neighbornodes[j]!=from){
      j++;
    }
    tosupp->outarcs[j]=&scndryarcs(tilenum,i);
  }

  /* set outputs */
  nscndrynodes[tilenum]=nnewnodes;
  nscndryarcs[tilenum]=nnewarcs;
  totarclens[tilenum]=totarclen;

  return(0);
}


/* function: FindNumPathsOut()
 * ---------------------------
 * Check all outgoing arcs to see how many paths out there are. 
 */
static
long FindNumPathsOut(nodeT *from, paramT *params, long tilerow, long tilecol,
                     long nnrow, long nncol, Array2D<short>& regions,
                     Array2D<short>& nextregions, Array2D<short>& lastregions,
                     Array2D<short>& regionsabove, Array2D<short>& regionsbelow, long prevncol){

  long npathsout, ntilerow, ntilecol, fromrow, fromcol;

  /* initialize */
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  fromrow=from->row;
  fromcol=from->col;
  npathsout=0;

  /* rightward arc */
  if(fromcol!=nncol-1){
    if(fromrow==0 || fromrow==nnrow-1
       || regions(fromrow-1,fromcol)!=regions(fromrow,fromcol)){
      npathsout++;
    }
  }else{
    if(fromrow==0 || fromrow==nnrow-1 ||
       (tilecol!=ntilecol-1
        && nextregions(fromrow-1,0)!=nextregions(fromrow,0))){
      npathsout++;
    }
  }

  /* downward arc */
  if(fromrow!=nnrow-1){
    if(fromcol==0 || fromcol==nncol-1
       || regions(fromrow,fromcol)!=regions(fromrow,fromcol-1)){
      npathsout++;
    }
  }else{
    if(fromcol==0 || fromcol==nncol-1 ||
       (tilerow!=ntilerow-1
        && regionsbelow(0,fromcol)!=regionsbelow(0,fromcol-1))){
      npathsout++;
    }
  }

  /* leftward arc */
  if(fromcol!=0){
    if(fromrow==0 || fromrow==nnrow-1
       || regions(fromrow,fromcol-1)!=regions(fromrow-1,fromcol-1)){
      npathsout++;
    }
  }else{
    if(fromrow==0 || fromrow==nnrow-1 ||
       (tilecol!=0
        && (lastregions(fromrow,prevncol-1)
            !=lastregions(fromrow-1,prevncol-1)))){
      npathsout++;
    }
  }

  /* upward arc */
  if(fromrow!=0){
    if(fromcol==0 || fromcol==nncol-1
       || regions(fromrow-1,fromcol-1)!=regions(fromrow-1,fromcol)){
      npathsout++;
    }
  }else{
    if(fromcol==0 || fromcol==nncol-1 ||
       (tilerow!=0
        && regionsabove(0,fromcol-1)!=regionsabove(0,fromcol))){
      npathsout++;
    }
  }

  /* return number of paths out of node */
  return(npathsout);

}


/* function: RegionTraceCheckNeighbors()
 * -------------------------------------
 */
template<class CostTag>
static
int RegionTraceCheckNeighbors(nodeT *from, nodeT **nextnodeptr,
                              Array2D<nodeT>& primarynodes, Array2D<short>& regions,
                              Array2D<short>& /*nextregions*/, Array2D<short>& /*lastregions*/,
                              Array2D<short>& /*regionsabove*/, Array2D<short>& /*regionsbelow*/,
                              long tilerow, long tilecol, long nnrow,
                              long nncol, Array2D<nodeT>& scndrynodes,
                              Array2D<nodesuppT>& nodesupp, Array2D<scndryarcT>& scndryarcs,
                              long *nnewnodesptr, long *nnewarcsptr,
                              long flowmax, long nrow, long ncol,
                              long prevnrow, long prevncol, paramT *params,
                              Array2D<typename CostTag::Cost>& costs, 
                              Array2D<typename CostTag::Cost>& rightedgecosts,
                              Array2D<typename CostTag::Cost>& loweredgecosts, 
                              Array2D<typename CostTag::Cost>& leftedgecosts,
                              Array2D<typename CostTag::Cost>& upperedgecosts, 
                              Array2D<short>& flows, Array2D<short>& rightedgeflows, 
                              Array2D<short>& loweredgeflows, Array2D<short>& leftedgeflows, 
                              Array2D<short>& upperedgeflows, 
                              Array2D<Array1D<long>>& scndrycosts,
                              Array1D<nodeT*>* updatednontilenodesptr,
                              long *nupdatednontilenodesptr,
                              long *updatednontilenodesizeptr,
                              Array1D<short>* inontilenodeoutarcptr,
                              long *totarclenptr, CostTag tag){

  long fromrow, fromcol;
  nodeT *to, *nextnode;

  /* initialize */
  fromrow=from->row;
  fromcol=from->col;
  nextnode=(*nextnodeptr);

  /* check rightward arc */
  if(fromcol!=nncol-1){
    to=&primarynodes(fromrow,fromcol+1);
    if(fromrow==0 || fromrow==nnrow-1
       || regions(fromrow-1,fromcol)!=regions(fromrow,fromcol)){
      if(to!=from->pred){
        to->pred=from;
        if(to->group==NOTINBUCKET){
          to->group=INBUCKET;
          to->next=nextnode;
          nextnode=to;
        }else if(to->group==ONTREE && (fromrow!=0 || tilerow==0)){
          TraceSecondaryArc(to,scndrynodes,nodesupp,scndryarcs,scndrycosts,
                            nnewnodesptr,nnewarcsptr,tilerow,tilecol,flowmax,
                            nrow,ncol,prevnrow,prevncol,params,costs,
                            rightedgecosts,loweredgecosts,leftedgecosts,
                            upperedgecosts,flows,rightedgeflows,
                            loweredgeflows,leftedgeflows,upperedgeflows,
                            updatednontilenodesptr,nupdatednontilenodesptr,
                            updatednontilenodesizeptr,inontilenodeoutarcptr,
                            totarclenptr,tag);
        }
      }
    }
  }

  /* check downward arc */
  if(fromrow!=nnrow-1){
    to=&primarynodes(fromrow+1,fromcol);
    if(fromcol==0 || fromcol==nncol-1
       || regions(fromrow,fromcol)!=regions(fromrow,fromcol-1)){
      if(to!=from->pred){
        to->pred=from;
        if(to->group==NOTINBUCKET){
          to->group=INBUCKET;
          to->next=nextnode;
          nextnode=to;
        }else if(to->group==ONTREE && (fromcol!=0 || tilecol==0)){
          TraceSecondaryArc(to,scndrynodes,nodesupp,scndryarcs,scndrycosts,
                            nnewnodesptr,nnewarcsptr,tilerow,tilecol,flowmax,
                            nrow,ncol,prevnrow,prevncol,params,costs,
                            rightedgecosts,loweredgecosts,leftedgecosts,
                            upperedgecosts,flows,rightedgeflows,
                            loweredgeflows,leftedgeflows,upperedgeflows,
                            updatednontilenodesptr,nupdatednontilenodesptr,
                            updatednontilenodesizeptr,inontilenodeoutarcptr,
                            totarclenptr,tag);
        }
      }
    }
  }

  /* check leftward arc */
  if(fromcol!=0){
    to=&primarynodes(fromrow,fromcol-1);
    if(fromrow==0 || fromrow==nnrow-1
       || regions(fromrow,fromcol-1)!=regions(fromrow-1,fromcol-1)){
      if(to!=from->pred){
        to->pred=from;
        if(to->group==NOTINBUCKET){
          to->group=INBUCKET;
          to->next=nextnode;
          nextnode=to;
        }else if(to->group==ONTREE && (fromrow!=0 || tilerow==0)){
          TraceSecondaryArc(to,scndrynodes,nodesupp,scndryarcs,scndrycosts,
                            nnewnodesptr,nnewarcsptr,tilerow,tilecol,flowmax,
                            nrow,ncol,prevnrow,prevncol,params,costs,
                            rightedgecosts,loweredgecosts,leftedgecosts,
                            upperedgecosts,flows,rightedgeflows,
                            loweredgeflows,leftedgeflows,upperedgeflows,
                            updatednontilenodesptr,nupdatednontilenodesptr,
                            updatednontilenodesizeptr,inontilenodeoutarcptr,
                            totarclenptr,tag);
        }
      }
    }
  }

  /* check upward arc */
  if(fromrow!=0){
    to=&primarynodes(fromrow-1,fromcol);
    if(fromcol==0 || fromcol==nncol-1
       || regions(fromrow-1,fromcol-1)!=regions(fromrow-1,fromcol)){
      if(to!=from->pred){
        to->pred=from;
        if(to->group==NOTINBUCKET){
          to->group=INBUCKET;
          to->next=nextnode;
          nextnode=to;
        }else if(to->group==ONTREE && (fromcol!=0 || tilecol==0)){
          TraceSecondaryArc(to,scndrynodes,nodesupp,scndryarcs,scndrycosts,
                            nnewnodesptr,nnewarcsptr,tilerow,tilecol,flowmax,
                            nrow,ncol,prevnrow,prevncol,params,costs,
                            rightedgecosts,loweredgecosts,leftedgecosts,
                            upperedgecosts,flows,rightedgeflows,
                            loweredgeflows,leftedgeflows,upperedgeflows,
                            updatednontilenodesptr,nupdatednontilenodesptr,
                            updatednontilenodesizeptr,inontilenodeoutarcptr,
                            totarclenptr,tag);
        }
      }
    }
  }

  /* set return values */
  *nextnodeptr=nextnode;
  return(0);
}


/* function: SetUpperEdge()
 * ------------------------
 */
template<class Cost>
static
int SetUpperEdge(long ncol, long tilerow, long tilecol, Array2D<Cost>& costs,
                 Array2D<Cost>& costsabove, Array2D<float>& unwphase,
                 Array2D<float>& unwphaseabove, Array2D<Cost>& upperedgecosts,
                 Array2D<short>& upperedgeflows, paramT *params, Array2D<short>& bulkoffsets){

  long col, reloffset;
  double dphi, dpsi;
  long nshortcycle;

  /* see if tile is in top row */
  if(tilerow!=0){

    /* set up */
    nshortcycle=params->nshortcycle;
    reloffset=bulkoffsets(tilerow-1,tilecol)-bulkoffsets(tilerow,tilecol);

    /* loop over all arcs on the boundary */
    for(col=0;col<ncol;col++){
      dphi=(unwphaseabove(0,col)-unwphase(0,col))/TWOPI;
      upperedgeflows(0,col)=(short )LRound(dphi)-reloffset;
      dpsi=dphi-floor(dphi);
      if(dpsi>0.5){
        dpsi-=1.0;
      }
      if constexpr(std::is_same<Cost,costT>{}){
        upperedgecosts(0,col).offset=(short )LRound(nshortcycle*dpsi);
        upperedgecosts(0,col).sigsq=AvgSigSq(costs(0,col).sigsq,
                                              costsabove(0,col).sigsq);
        if(costs(0,col).dzmax>costsabove(0,col).dzmax){
          upperedgecosts(0,col).dzmax=costs(0,col).dzmax;
        }else{
          upperedgecosts(0,col).dzmax=costsabove(0,col).dzmax;
        }
        if(costs(0,col).laycost<costsabove(0,col).laycost){
          upperedgecosts(0,col).laycost=costs(0,col).laycost;
        }else{
          upperedgecosts(0,col).laycost=costsabove(0,col).laycost;
        }
      }else if constexpr(std::is_same<Cost,smoothcostT>{}){
        upperedgecosts(0,col).offset=(short )LRound(nshortcycle*dpsi);
        upperedgecosts(0,col).sigsq
          =AvgSigSq(costs(0,col).sigsq,costsabove(0,col).sigsq);
      }else if constexpr(std::is_same<Cost,bidircostT>{}){
        upperedgecosts(0,col).posweight=
          (costs(0,col).posweight
           +costsabove(0,col).posweight)/2;
        upperedgecosts(0,col).negweight=
          (costs(0,col).negweight
           +costsabove(0,col).negweight)/2;
      }else if constexpr(std::is_same<Cost,short>{}){
        upperedgecosts(0,col)=
          (costs(0,col)+costsabove(0,col))/2;
      }else{
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Illegal cost mode in SetUpperEdge(). This is a bug.");
      }
    }
  }else{
    if constexpr(std::is_same<Cost,costT>{}){
      for(col=0;col<ncol;col++){
        upperedgecosts(0,col).offset=LARGESHORT/2;
        upperedgecosts(0,col).sigsq=LARGESHORT;
        upperedgecosts(0,col).dzmax=LARGESHORT;
        upperedgecosts(0,col).laycost=0;
      }
    }else if constexpr(std::is_same<Cost,smoothcostT>{}){
      for(col=0;col<ncol;col++){
        upperedgecosts(0,col).offset=0;
        upperedgecosts(0,col).sigsq=LARGESHORT;
      }
    }else if constexpr(std::is_same<Cost,bidircostT>{}){
      for(col=0;col<ncol;col++){
        upperedgecosts(0,col).posweight=0;
        upperedgecosts(0,col).negweight=0;
      }
    }else if constexpr(std::is_same<Cost,short>{}){
      for(col=0;col<ncol;col++){
        upperedgecosts(0,col)=0;
      }
    }else{
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Illegal cost mode in SetUpperEdge(). This is a bug.");
    }
  }

  /* done */
  return(0);
  
}


/* function: SetLowerEdge()
 * ------------------------
 */
template<class Cost>
static
int SetLowerEdge(long nrow, long ncol, long tilerow, long tilecol,
                 Array2D<Cost>& costs, Array2D<Cost>& costsbelow,
                 Array2D<float>& unwphase, Array2D<float>& unwphasebelow,
                 Array2D<Cost>& loweredgecosts, Array2D<short>& loweredgeflows,
                 paramT *params, Array2D<short>& bulkoffsets){

  long col, iflow, reloffset, nmax;
  long flowlimhi, flowlimlo, maxflow, minflow, tempflow;
  double dphi, dpsi;
  long nshortcycle;

  /* see if tile is in bottom row */
  if(tilerow!=params->ntilerow-1){
  
    /* set up */
    nshortcycle=params->nshortcycle;
    flowlimhi=LARGESHORT;
    flowlimlo=-LARGESHORT;
    minflow=flowlimhi;
    maxflow=flowlimlo;

    Array1D<long> flowhistogram=Array1D<long>::Zero(flowlimhi-flowlimlo+1);

    /* loop over all arcs on the boundary */
    for(col=0;col<ncol;col++){
      dphi=(unwphase(nrow-1,col)-unwphasebelow(0,col))/TWOPI;
      tempflow=(short )LRound(dphi);
      loweredgeflows(0,col)=tempflow;
      if(tempflow<minflow){
        if(tempflow<flowlimlo){
          fflush(NULL);
          throw isce3::except::OverflowError(ISCE_SRCINFO(),
                  "Overflow in tile offset");
        }
        minflow=tempflow;
      }
      if(tempflow>maxflow){
        if(tempflow>flowlimhi){
          fflush(NULL);
          throw isce3::except::OverflowError(ISCE_SRCINFO(),
                  "Overflow in tile offset");
        }
        maxflow=tempflow;
      }
      flowhistogram[tempflow-flowlimlo]++;
      dpsi=dphi-floor(dphi);
      if(dpsi>0.5){
        dpsi-=1.0;
      }
      if constexpr(std::is_same<Cost,costT>{}){
        loweredgecosts(0,col).offset=(short )LRound(nshortcycle*dpsi);
        loweredgecosts(0,col).sigsq=AvgSigSq(costs(nrow-2,col).sigsq,
                                              costsbelow(0,col).sigsq);
        if(costs(nrow-2,col).dzmax>costsbelow(0,col).dzmax){
          loweredgecosts(0,col).dzmax=costs(nrow-2,col).dzmax;
        }else{
          loweredgecosts(0,col).dzmax=costsbelow(0,col).dzmax;
        }
        if(costs(nrow-2,col).laycost<costsbelow(0,col).laycost){
          loweredgecosts(0,col).laycost=costs(nrow-2,col).laycost;
        }else{
          loweredgecosts(0,col).laycost=costsbelow(0,col).laycost;
        }
      }else if constexpr(std::is_same<Cost,smoothcostT>{}){
        loweredgecosts(0,col).offset=(short )LRound(nshortcycle*dpsi);
        loweredgecosts(0,col).sigsq
          =AvgSigSq(costs(nrow-2,col).sigsq,costsbelow(0,col).sigsq);
      }else if constexpr(std::is_same<Cost,bidircostT>{}){
        loweredgecosts(0,col).posweight=
          (costs(nrow-2,col).posweight
           +costsbelow(0,col).posweight)/2;
        loweredgecosts(0,col).negweight=
          (costs(nrow-2,col).negweight
           +costsbelow(0,col).negweight)/2;
      }else if constexpr(std::is_same<Cost,short>{}){
        loweredgecosts(0,col)=
          (costs(nrow-2,col)
           +costsbelow(0,col))/2;
      }else{
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Illegal cost mode in SetLowerEdge(). This is a bug.");
      }
    }

    /* set bulk tile offset equal to mode of flow histogram */
    nmax=0;
    reloffset=0;
    for(iflow=minflow;iflow<=maxflow;iflow++){
      if(flowhistogram[iflow-flowlimlo]>nmax){
        nmax=flowhistogram[iflow-flowlimlo];
        reloffset=iflow;
      }
    }
    bulkoffsets(tilerow+1,tilecol)=bulkoffsets(tilerow,tilecol)-reloffset;

    /* subtract relative tile offset from edge flows */
    for(col=0;col<ncol;col++){
      loweredgeflows(0,col)-=reloffset;
    }

  }else{
    if constexpr(std::is_same<Cost,costT>{}){
      for(col=0;col<ncol;col++){
        loweredgecosts(0,col).offset=LARGESHORT/2;
        loweredgecosts(0,col).sigsq=LARGESHORT;
        loweredgecosts(0,col).dzmax=LARGESHORT;
        loweredgecosts(0,col).laycost=0;
      }
    }else if constexpr(std::is_same<Cost,smoothcostT>{}){
      for(col=0;col<ncol;col++){
        loweredgecosts(0,col).offset=0;
        loweredgecosts(0,col).sigsq=LARGESHORT;
      }
    }else if constexpr(std::is_same<Cost,bidircostT>{}){
      for(col=0;col<ncol;col++){
        loweredgecosts(0,col).posweight=0;
        loweredgecosts(0,col).negweight=0;
      }
    }else if constexpr(std::is_same<Cost,short>{}){
      for(col=0;col<ncol;col++){
        loweredgecosts(0,col)=0;
      }
    }else{
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Illegal cost mode in SetLowerEdge(). This is a bug.");
    }
  }

  /* done */
  return(0);

}


/* function: SetLeftEdge()
 * -----------------------
 */
template<class Cost>
static
int SetLeftEdge(long nrow, long prevncol, long tilerow, long tilecol,
                Array2D<Cost>& costs, Array2D<Cost>& lastcosts, Array2D<float>& unwphase,
                Array2D<float>& lastunwphase, Array2D<Cost>& leftedgecosts,
                Array2D<short>& leftedgeflows, paramT *params, Array2D<short>& bulkoffsets){

  long row, reloffset;
  double dphi, dpsi;
  long nshortcycle;

  /* see if tile is in left column */
  if(tilecol!=0){

    /* set up */
    nshortcycle=params->nshortcycle;
    reloffset=bulkoffsets(tilerow,tilecol)-bulkoffsets(tilerow,tilecol-1);

    /* loop over all arcs on the boundary */
    for(row=0;row<nrow;row++){
      dphi=(unwphase(row,0)
            -lastunwphase(row,prevncol-1))/TWOPI;
      leftedgeflows(row,0)=(short )LRound(dphi)-reloffset;
      dpsi=dphi-floor(dphi);
      if(dpsi>0.5){
        dpsi-=1.0;
      }
      if constexpr(std::is_same<Cost,costT>{}){
        leftedgecosts(row,0).offset=(short )LRound(TILEDPSICOLFACTOR
                                                    *nshortcycle*dpsi);
        leftedgecosts(row,0).sigsq
          =AvgSigSq(costs(row+nrow-1,0).sigsq,
                    lastcosts(row+nrow-1,prevncol-2).sigsq);
        if(costs(row+nrow-1,0).dzmax>lastcosts(row+nrow-1,prevncol-2).dzmax){
          leftedgecosts(row,0).dzmax=costs(row+nrow-1,0).dzmax;
        }else{
          leftedgecosts(row,0).dzmax=lastcosts(row+nrow-1,prevncol-2).dzmax;
        }
        if(costs(row+nrow-1,0).laycost
           >lastcosts(row+nrow-1,prevncol-2).laycost){
          leftedgecosts(row,0).laycost=costs(row+nrow-1,0).laycost;
        }else{
          leftedgecosts(row,0).laycost
            =lastcosts(row+nrow-1,prevncol-2).laycost;
        }
      }else if constexpr(std::is_same<Cost,smoothcostT>{}){
        leftedgecosts(row,0).offset
          =(short )LRound(TILEDPSICOLFACTOR*nshortcycle*dpsi);
        leftedgecosts(row,0).sigsq
          =AvgSigSq(costs(row+nrow-1,0).sigsq,
                    lastcosts(row+nrow-1,prevncol-2).sigsq);
      }else if constexpr(std::is_same<Cost,bidircostT>{}){
        leftedgecosts(row,0).posweight=
          (costs(row+nrow-1,0).posweight
           +lastcosts(row+nrow-1,prevncol-2).posweight)
          /2;
        leftedgecosts(row,0).negweight=
          (costs(row+nrow-1,0).negweight
           +lastcosts(row+nrow-1,prevncol-2).negweight)
          /2;
      }else if constexpr(std::is_same<Cost,short>{}){
        leftedgecosts(row,0)=
          (costs(row+nrow-1,0)
           +lastcosts(row+nrow-1,prevncol-2))/2;
      }else{
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Illegal cost mode in SetLeftEdge(). This is a bug.");
      }
    }
  }else{
    if constexpr(std::is_same<Cost,costT>{}){
      for(row=0;row<nrow;row++){
        leftedgecosts(row,0).offset=LARGESHORT/2;
        leftedgecosts(row,0).sigsq=LARGESHORT;
        leftedgecosts(row,0).dzmax=LARGESHORT;
        leftedgecosts(row,0).laycost=0;
      }
    }else if constexpr(std::is_same<Cost,smoothcostT>{}){
      for(row=0;row<nrow;row++){
        leftedgecosts(row,0).offset=0;
        leftedgecosts(row,0).sigsq=LARGESHORT;
      }
    }else if constexpr(std::is_same<Cost,bidircostT>{}){
      for(row=0;row<nrow;row++){
        leftedgecosts(row,0).posweight=0;
        leftedgecosts(row,0).negweight=0;
      }
    }else if constexpr(std::is_same<Cost,short>{}){
      for(row=0;row<nrow;row++){
        leftedgecosts(row,0)=0;
      }
    }else{
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Illegal cost mode in SetLeftEdge(). This is a bug.");
    }
  }

  /* done */
  return(0);

}


/* function: SetRightEdge()
 * ------------------------
 */
template<class Cost>
static
int SetRightEdge(long nrow, long ncol, long tilerow, long tilecol,
                 Array2D<Cost>& costs, Array2D<Cost>& nextcosts,
                 Array2D<float>& unwphase, Array2D<float>& nextunwphase,
                 Array2D<Cost>& rightedgecosts, Array2D<short>& rightedgeflows,
                 paramT *params, Array2D<short>& bulkoffsets){

  long row, iflow, reloffset, nmax;
  long flowlimhi, flowlimlo, maxflow, minflow, tempflow;
  double dphi, dpsi;
  long nshortcycle;

  /* see if tile in right column */  
  if(tilecol!=params->ntilecol-1){

    /* set up */
    nshortcycle=params->nshortcycle;
    flowlimhi=LARGESHORT;
    flowlimlo=-LARGESHORT;
    minflow=flowlimhi;
    maxflow=flowlimlo;

    Array1D<long> flowhistogram=Array1D<long>::Zero(flowlimhi-flowlimlo+1);

    /* loop over all arcs on the boundary */
    for(row=0;row<nrow;row++){
      dphi=(nextunwphase(row,0)
            -unwphase(row,ncol-1))/TWOPI;
      tempflow=(short )LRound(dphi);
      rightedgeflows(row,0)=tempflow;
      if(tempflow<minflow){
        if(tempflow<flowlimlo){
          fflush(NULL);
          throw isce3::except::OverflowError(ISCE_SRCINFO(),
                  "Overflow in tile offset");
        }
        minflow=tempflow;
      }
      if(tempflow>maxflow){
        if(tempflow>flowlimhi){
          fflush(NULL);
          throw isce3::except::OverflowError(ISCE_SRCINFO(),
                  "Overflow in tile offset");
        }
        maxflow=tempflow;
      }
      flowhistogram[tempflow-flowlimlo]++;    
      dpsi=dphi-floor(dphi);
      if(dpsi>0.5){
        dpsi-=1.0;
      }
      if constexpr(std::is_same<Cost,costT>{}){
        rightedgecosts(row,0).offset=(short )LRound(TILEDPSICOLFACTOR
                                                     *nshortcycle*dpsi);
        rightedgecosts(row,0).sigsq
          =AvgSigSq(costs(row+nrow-1,ncol-2).sigsq,
                    nextcosts(row+nrow-1,0).sigsq);
        if(costs(row+nrow-1,ncol-2).dzmax>nextcosts(row+nrow-1,0).dzmax){
          rightedgecosts(row,0).dzmax=costs(row+nrow-1,ncol-2).dzmax;
        }else{
          rightedgecosts(row,0).dzmax=nextcosts(row+nrow-1,0).dzmax;
        }
        if(costs(row+nrow-1,ncol-2).laycost>nextcosts(row+nrow-1,0).laycost){
          rightedgecosts(row,0).laycost=costs(row+nrow-1,ncol-2).laycost;
        }else{
          rightedgecosts(row,0).laycost=nextcosts(row+nrow-1,0).laycost;
        }
      }else if constexpr(std::is_same<Cost,smoothcostT>{}){
        rightedgecosts(row,0).offset
          =(short )LRound(TILEDPSICOLFACTOR*nshortcycle*dpsi);
        rightedgecosts(row,0).sigsq
          =AvgSigSq(costs(row+nrow-1,ncol-2).sigsq,
                    nextcosts(row+nrow-1,0).sigsq);
      }else if constexpr(std::is_same<Cost,bidircostT>{}){
        rightedgecosts(row,0).posweight=
          (costs(row+nrow-1,ncol-2).posweight
           +nextcosts(row+nrow-1,ncol-2).posweight)/2;
        rightedgecosts(row,0).negweight=
          (costs(row+nrow-1,ncol-2).negweight
           +nextcosts(row+nrow-1,ncol-2).negweight)/2;
      }else if constexpr(std::is_same<Cost,short>{}){
        rightedgecosts(row,0)=
          (costs(row+nrow-1,ncol-2)
           +nextcosts(row+nrow-1,0))/2;
      }else{
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Illegal cost mode in SetRightEdge(). This is a bug.");
      }
    }

    /* set bulk tile offset equal to mode of flow histogram */
    if(tilerow==0){
      nmax=0;
      reloffset=0;
      for(iflow=minflow;iflow<=maxflow;iflow++){
        if(flowhistogram[iflow-flowlimlo]>nmax){
          nmax=flowhistogram[iflow-flowlimlo];
          reloffset=iflow;
        }
      }
      bulkoffsets(tilerow,tilecol+1)=bulkoffsets(tilerow,tilecol)+reloffset;
    }else{
      reloffset=bulkoffsets(tilerow,tilecol+1)-bulkoffsets(tilerow,tilecol);
    }

    /* subtract relative tile offset from edge flows */
    for(row=0;row<nrow;row++){
      rightedgeflows(row,0)-=reloffset;
    }

  }else{
    if constexpr(std::is_same<Cost,costT>{}){
      for(row=0;row<nrow;row++){
        rightedgecosts(row,0).offset=LARGESHORT/2;
        rightedgecosts(row,0).sigsq=LARGESHORT;
        rightedgecosts(row,0).dzmax=LARGESHORT;
        rightedgecosts(row,0).laycost=0;
      }
    }else if constexpr(std::is_same<Cost,smoothcostT>{}){
      for(row=0;row<nrow;row++){
        rightedgecosts(row,0).offset=0;
        rightedgecosts(row,0).sigsq=LARGESHORT;
      }
    }else if constexpr(std::is_same<Cost,bidircostT>{}){
      for(row=0;row<nrow;row++){
        rightedgecosts(row,0).posweight=0;
        rightedgecosts(row,0).negweight=0;
      }
    }else if constexpr(std::is_same<Cost,short>{}){
      for(row=0;row<nrow;row++){
        rightedgecosts(row,0)=0;
      }
    }else{
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Illegal cost mode in SetRightEdge(). This is a bug.");
    }
  }

  /* done */
  return(0);

}


/* function: AvgSigSq()
 * --------------------
 * Return average of sigsq values after chcking for special value and
 * clipping to short.
 */
static
short AvgSigSq(short sigsq1, short sigsq2){

  int sigsqavg;

  
  /* if either value is special LARGESHORT value, use that */
  if(sigsq1==LARGESHORT || sigsq2==LARGESHORT){
    return(LARGESHORT);
  }

  /* compute average */
  sigsqavg=(int )ceil(0.5*(((int )sigsq1)+((int )sigsq2)));

  /* clip */
  sigsqavg=LClip(sigsqavg,-LARGESHORT,LARGESHORT);

  /* return */
  return((short )sigsqavg);
  
}


/* function: TraceSecondaryArc()
 * -----------------------------
 */
template<class CostTag>
static
int TraceSecondaryArc(nodeT *primaryhead, Array2D<nodeT>& scndrynodes,
                      Array2D<nodesuppT>& nodesupp, Array2D<scndryarcT>& scndryarcs,
                      Array2D<Array1D<long>>& scndrycosts, long *nnewnodesptr,
                      long *nnewarcsptr, long tilerow, long tilecol,
                      long flowmax, long nrow, long ncol,
                      long prevnrow, long prevncol, paramT *params,
                      Array2D<typename CostTag::Cost>& tilecosts, 
                      Array2D<typename CostTag::Cost>& rightedgecosts,
                      Array2D<typename CostTag::Cost>& loweredgecosts, 
                      Array2D<typename CostTag::Cost>& leftedgecosts,
                      Array2D<typename CostTag::Cost>& upperedgecosts, 
                      Array2D<short>& tileflows, Array2D<short>& rightedgeflows, 
                      Array2D<short>& loweredgeflows, Array2D<short>& leftedgeflows, 
                      Array2D<short>& upperedgeflows, Array1D<nodeT*>* updatednontilenodesptr,
                      long *nupdatednontilenodesptr, long *updatednontilenodesizeptr,
                      Array1D<short>* inontilenodeoutarcptr, long *totarclenptr, CostTag tag){

  long i, row, col, nnewnodes, arclen, ntilerow, ntilecol, arcnum;
  long tilenum, nflow, primaryarcrow, primaryarccol, poscost, negcost, nomcost;
  long nnrow, nncol, calccostnrow, nnewarcs, arroffset, nshortcycle;
  long mincost, mincostflow, minweight, maxcost;
  double sigsq, sumsigsqinv, tempdouble, tileedgearcweight;
  nodeT *tempnode, *primarytail, *scndrytail, *scndryhead;
  nodeT *primarydummy, *scndrydummy;
  nodesuppT *supptail, *supphead, *suppdummy;
  scndryarcT *newarc;
  signed char primaryarcdir, zerocost;

  Array2D<short>* flowsptr=nullptr;

  using Cost=typename CostTag::Cost;
  Array2D<Cost>* costsptr=nullptr;

  /* do nothing if source is passed or if arc already done in previous tile */
  if(primaryhead->pred==NULL
     || (tilerow!=0 && primaryhead->row==0 && primaryhead->pred->row==0)
     || (tilecol!=0 && primaryhead->col==0 && primaryhead->pred->col==0)){
    return(0);
  }

  /* set up */
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  nnrow=nrow+1;
  nncol=ncol+1;
  tilenum=tilerow*ntilecol+tilecol;
  tileedgearcweight=params->tileedgeweight;
  nshortcycle=params->nshortcycle;
  zerocost=FALSE;
  arroffset=0;
  sigsq=0;

  auto scndrycostarr=Array1D<long>(2*flowmax+2);

  /* loop to determine appropriate value for arroffset */
  while(TRUE){

    /* initialize variables */
    arclen=0;
    sumsigsqinv=0;
    for(nflow=1;nflow<=2*flowmax;nflow++){
      scndrycostarr[nflow]=0;
    }

    /* loop over primary arcs on secondary arc again to get costs */
    primarytail=primaryhead->pred;
    tempnode=primaryhead;
    while(TRUE){

      /* get primary arc just traversed */
      arclen++;
      if(tempnode->col==primarytail->col+1){              /* rightward arc */
        primaryarcdir=1;
        primaryarccol=primarytail->col;
        if(primarytail->row==0){                               /* top edge */
          if(tilerow==0){
            zerocost=TRUE;
          }else{
            primaryarcrow=0;
            costsptr=&upperedgecosts;
            flowsptr=&upperedgeflows;
            calccostnrow=2;
          }
        }else if(primarytail->row==nnrow-1){                /* bottom edge */
          if(tilerow==ntilerow-1){
            zerocost=TRUE;
          }else{
            primaryarcrow=0;
            costsptr=&loweredgecosts;
            flowsptr=&loweredgeflows;
            calccostnrow=2;
          }
        }else{                                               /* normal arc */
          primaryarcrow=primarytail->row-1;
          costsptr=&tilecosts;
          flowsptr=&tileflows;
          calccostnrow=nrow;
        }
      }else if(tempnode->row==primarytail->row+1){         /* downward arc */
        primaryarcdir=1;
        if(primarytail->col==0){                              /* left edge */
          if(tilecol==0){
            zerocost=TRUE;
          }else{
            primaryarcrow=primarytail->row;
            primaryarccol=0;
            costsptr=&leftedgecosts;
            flowsptr=&leftedgeflows;
            calccostnrow=0;
          }
        }else if(primarytail->col==nncol-1){                 /* right edge */
          if(tilecol==ntilecol-1){
            zerocost=TRUE;
          }else{
            primaryarcrow=primarytail->row;
            primaryarccol=0;
            costsptr=&rightedgecosts;
            flowsptr=&rightedgeflows;
            calccostnrow=0;
          }
        }else{                                               /* normal arc */
          primaryarcrow=primarytail->row+nrow-1;
          primaryarccol=primarytail->col-1;
          costsptr=&tilecosts;
          flowsptr=&tileflows;
          calccostnrow=nrow;
        }
      }else if(tempnode->col==primarytail->col-1){         /* leftward arc */
        primaryarcdir=-1;
        primaryarccol=primarytail->col-1;
        if(primarytail->row==0){                               /* top edge */
          if(tilerow==0){
            zerocost=TRUE;
          }else{
            primaryarcrow=0;
            costsptr=&upperedgecosts;
            flowsptr=&upperedgeflows;
            calccostnrow=2;
          }
        }else if(primarytail->row==nnrow-1){                /* bottom edge */
          if(tilerow==ntilerow-1){
            zerocost=TRUE;
          }else{
            primaryarcrow=0;
            costsptr=&loweredgecosts;
            flowsptr=&loweredgeflows;
            calccostnrow=2;
          }
        }else{                                               /* normal arc */
          primaryarcrow=primarytail->row-1;
          costsptr=&tilecosts;
          flowsptr=&tileflows;
          calccostnrow=nrow;
        }
      }else{                                                 /* upward arc */
        primaryarcdir=-1;
        if(primarytail->col==0){                              /* left edge */
          if(tilecol==0){
            zerocost=TRUE;
          }else{
            primaryarcrow=primarytail->row-1;
            primaryarccol=0;
            costsptr=&leftedgecosts;
            flowsptr=&leftedgeflows;
            calccostnrow=0;
          }
        }else if(primarytail->col==nncol-1){                 /* right edge */
          if(tilecol==ntilecol-1){
            zerocost=TRUE;
          }else{            
            primaryarcrow=primarytail->row-1;
            primaryarccol=0;
            costsptr=&rightedgecosts;
            flowsptr=&rightedgeflows;
            calccostnrow=0;
          }
        }else{                                               /* normal arc */
          primaryarcrow=primarytail->row+nrow-2;
          primaryarccol=primarytail->col-1;
          costsptr=&tilecosts;
          flowsptr=&tileflows;
          calccostnrow=nrow;
        }
      }

      /* keep absolute cost of arc to the previous node */
      if(!zerocost){

        /* accumulate incremental cost in table for each nflow increment */
        /* offset flow in flow array temporarily by arroffset then undo below */
        (*flowsptr)(primaryarcrow,primaryarccol)-=primaryarcdir*arroffset;
        nomcost=EvalCost(*costsptr,*flowsptr,primaryarcrow,primaryarccol,calccostnrow,
                         params,tag);
        for(nflow=1;nflow<=flowmax;nflow++){
          (*flowsptr)(primaryarcrow,primaryarccol)+=(primaryarcdir*nflow);
          poscost=EvalCost(*costsptr,*flowsptr,primaryarcrow,primaryarccol,
                           calccostnrow,params,tag);
          (*flowsptr)(primaryarcrow,primaryarccol)-=(2*primaryarcdir*nflow);
          negcost=EvalCost(*costsptr,*flowsptr,primaryarcrow,primaryarccol,
                           calccostnrow,params,tag);
          (*flowsptr)(primaryarcrow,primaryarccol)+=(primaryarcdir*nflow);
          tempdouble=(scndrycostarr[nflow]+(poscost-nomcost));
          if(tempdouble>LARGEINT){
            scndrycostarr[nflow]=LARGEINT;
          }else if(tempdouble<-LARGEINT){
            scndrycostarr[nflow]=-LARGEINT;
          }else{
            scndrycostarr[nflow]+=(poscost-nomcost);
          }
          tempdouble=(scndrycostarr[nflow+flowmax]+(negcost-nomcost));
          if(tempdouble>LARGEINT){
            scndrycostarr[nflow+flowmax]=LARGEINT;
          }else if(tempdouble<-LARGEINT){
            scndrycostarr[nflow+flowmax]=-LARGEINT;
          }else{
            scndrycostarr[nflow+flowmax]+=(negcost-nomcost);
          }
        }
        (*flowsptr)(primaryarcrow,primaryarccol)+=primaryarcdir*arroffset;

        /* accumulate term to be used for cost growth beyond table bounds */
        if constexpr(std::is_same<CostTag,TopoCostTag>{} || std::is_same<CostTag,DefoCostTag>{}){
          sigsq=(*costsptr)(primaryarcrow,primaryarccol).sigsq;
        }else if constexpr(std::is_same<CostTag,SmoothCostTag>{}){
          sigsq=(*costsptr)(primaryarcrow,primaryarccol).sigsq;
        }else if constexpr(std::is_same<CostTag,L0CostTag>{} || std::is_same<CostTag,L1CostTag>{}
                 || std::is_same<CostTag,L2CostTag>{} || std::is_same<CostTag,LPCostTag>{}){
          minweight=(*costsptr)(primaryarcrow,primaryarccol);
          if(minweight<1){
            sigsq=LARGESHORT;
          }else{
            sigsq=1.0/(double )minweight;
          }
        }else if constexpr(std::is_same<CostTag,L0BiDirCostTag>{} || std::is_same<CostTag,L1BiDirCostTag>{}
                 || std::is_same<CostTag,L2BiDirCostTag>{} || std::is_same<CostTag,LPBiDirCostTag>{}){
          minweight=LMin((*costsptr)(primaryarcrow,primaryarccol)
                         .posweight,
                         (*costsptr)(primaryarcrow,primaryarccol)
                         .negweight);
          if(minweight<1){
            sigsq=LARGESHORT;
          }else{
            sigsq=1.0/(double )minweight;
          }
        }
        if(sigsq<LARGESHORT){    /* zero cost arc if sigsq == LARGESHORT */
          sumsigsqinv+=(1.0/sigsq);
        }
      }

      /* break if found the secondary arc tail */
      if(primarytail->group==ONTREE){
        break;
      }  

      /* move up the tree */
      tempnode=primarytail;
      primarytail=primarytail->pred;

    } /* end while loop for tracing secondary arc for costs */

    /* break if we have a zero-cost arc on the edge of the full array */
    if(zerocost){
      break;
    }

    /* find flow index with minimum cost */
    mincost=0;
    maxcost=0;
    mincostflow=0;
    for(nflow=1;nflow<=flowmax;nflow++){
      if(scndrycostarr[nflow]<mincost){
        mincost=scndrycostarr[nflow];
        mincostflow=nflow;
      }
      if(scndrycostarr[flowmax+nflow]<mincost){
        mincost=scndrycostarr[flowmax+nflow];
        mincostflow=-nflow;
      }
      if(scndrycostarr[nflow]>maxcost){
        maxcost=scndrycostarr[nflow];
      }
      if(scndrycostarr[flowmax+nflow]>maxcost){
        maxcost=scndrycostarr[flowmax+nflow];
      }
    }

    /* if cost was all zero, treat as zero cost arc */
    if(maxcost==mincost){
      zerocost=TRUE;
      sumsigsqinv=0;
    }

    /* break if cost array adequately centered on minimum cost flow */
    if(mincostflow==0){
      break;
    }

    /* correct value of arroffset for next loop */
    if(mincostflow==flowmax){
      arroffset-=((long )floor(1.5*flowmax));
    }else if(mincostflow==-flowmax){
      arroffset+=((long )floor(1.5*flowmax));      
    }else{
      arroffset-=mincostflow;
    }

  } /* end while loop for determining arroffset */

  /* ignore this arc if primary head is same as tail (ie, if arc loops) */
  /* only way this can happen is if region is connected at one corner only */
  /* so any possible improvements should have been made by primary solver */
  if(primaryhead==primarytail){
    scndrycostarr=Array1D<long>{};
    return(0);
  }

  /* see if we have a secondary arc on the edge of the full-sized array */
  /* these arcs have zero cost since the edge is treated as a single node */
  /* secondary arcs whose primary arcs all have zero cost are also zeroed */
  if(zerocost){

    /* set sum of standard deviations to indicate zero-cost secondary arc */
    scndrycostarr[0]=0;
    for(nflow=1;nflow<=2*flowmax;nflow++){
      scndrycostarr[nflow]=0;
    }
    scndrycostarr[2*flowmax+1]=ZEROCOSTARC;

  }else{

    /* give extra weight to arcs on tile edges */
    if((primaryhead->row==primarytail->row 
        && (primaryhead->row==0 || primaryhead->row==nnrow-1))
       || (primaryhead->col==primarytail->col
           && (primaryhead->col==0 || primaryhead->col==nncol-1))){
      for(nflow=1;nflow<=2*flowmax;nflow++){
        tempdouble=scndrycostarr[nflow]*tileedgearcweight;
        if(tempdouble>LARGEINT){
          scndrycostarr[nflow]=LARGEINT;
        }else if(tempdouble<-LARGEINT){
          scndrycostarr[nflow]=-LARGEINT;
        }else{
          scndrycostarr[nflow]=LRound(tempdouble);
        }
      }
      sumsigsqinv*=tileedgearcweight; 

    }

    /* store sum of primary cost variances at end of secondary cost array */
    tempdouble=sumsigsqinv*nshortcycle*nshortcycle;
    if(tempdouble<LARGEINT){
      scndrycostarr[2*flowmax+1]=LRound(tempdouble);
    }else{
      scndrycostarr[2*flowmax+1]=LARGEINT;
    }
    scndrycostarr[0]=arroffset;

  }

  /* find secondary nodes corresponding to primary head, tail */
  if(primarytail->row==0 && tilerow!=0){
    scndrytail=FindScndryNode(scndrynodes,nodesupp,
                              (tilerow-1)*ntilecol+tilecol,
                              prevnrow,primarytail->col);
  }else if(primarytail->col==0 && tilecol!=0){
    scndrytail=FindScndryNode(scndrynodes,nodesupp,
                              tilerow*ntilecol+(tilecol-1),
                              primarytail->row,prevncol);
  }else{
    scndrytail=FindScndryNode(scndrynodes,nodesupp,tilenum,
                              primarytail->row,primarytail->col);
  }
  if(primaryhead->row==0 && tilerow!=0){
    scndryhead=FindScndryNode(scndrynodes,nodesupp,
                              (tilerow-1)*ntilecol+tilecol,
                              prevnrow,primaryhead->col);
  }else if(primaryhead->col==0 && tilecol!=0){
    scndryhead=FindScndryNode(scndrynodes,nodesupp,
                              tilerow*ntilecol+(tilecol-1),
                              primaryhead->row,prevncol);
  }else{
    scndryhead=FindScndryNode(scndrynodes,nodesupp,tilenum,
                              primaryhead->row,primaryhead->col);
  }

  /* see if there is already arc between secondary head, tail */
  row=scndrytail->row;
  col=scndrytail->col;
  for(i=0;i<nodesupp(row,col).noutarcs;i++){
    tempnode=nodesupp(row,col).neighbornodes[i];
    if((nodesupp(row,col).outarcs[i]==NULL
        && tempnode->row==primaryhead->row
        && tempnode->col==primaryhead->col)
       || (nodesupp(row,col).outarcs[i]!=NULL
           && tempnode->row==scndryhead->row
           && tempnode->col==scndryhead->col)){

      /* see if secondary arc traverses only one primary arc */
      primarydummy=primaryhead->pred;
      if(primarydummy->group!=ONTREE){
      
        /* arc already exists, free memory for cost array (will trace again) */
        scndrycostarr=Array1D<long>{};

        /* set up dummy node */
        primarydummy->group=ONTREE;
        nnewnodes=++(*nnewnodesptr);
        if(nnewnodes>scndrynodes.cols()){
          auto nnewcols=std::max(nnewnodes,2*scndrynodes.cols());
          scndrynodes.conservativeResize(Eigen::NoChange,nnewcols);
        }
        scndrydummy=&scndrynodes(tilenum,nnewnodes-1);
        if(nnewnodes>nodesupp.cols()){
          auto nnewcols=std::max(nnewnodes,2*nodesupp.cols());
          nodesupp.conservativeResize(Eigen::NoChange,nnewcols);
        }
        suppdummy=&nodesupp(tilenum,nnewnodes-1);
        scndrydummy->row=tilenum;
        scndrydummy->col=nnewnodes-1;
        suppdummy->row=primarydummy->row;
        suppdummy->col=primarydummy->col;
        suppdummy->noutarcs=0;
        suppdummy->neighbornodes = {};
        suppdummy->outarcs = {};

        /* recursively call TraceSecondaryArc() to set up arcs */
        TraceSecondaryArc(primarydummy,scndrynodes,nodesupp,scndryarcs,
                          scndrycosts,nnewnodesptr,nnewarcsptr,tilerow,tilecol,
                          flowmax,nrow,ncol,prevnrow,prevncol,params,tilecosts,
                          rightedgecosts,loweredgecosts,leftedgecosts,
                          upperedgecosts,tileflows,rightedgeflows,
                          loweredgeflows,leftedgeflows,upperedgeflows,
                          updatednontilenodesptr,nupdatednontilenodesptr,
                          updatednontilenodesizeptr,inontilenodeoutarcptr,
                          totarclenptr,tag);
        TraceSecondaryArc(primaryhead,scndrynodes,nodesupp,scndryarcs,
                          scndrycosts,nnewnodesptr,nnewarcsptr,tilerow,tilecol,
                          flowmax,nrow,ncol,prevnrow,prevncol,params,tilecosts,
                          rightedgecosts,loweredgecosts,leftedgecosts,
                          upperedgecosts,tileflows,rightedgeflows,
                          loweredgeflows,leftedgeflows,upperedgeflows,
                          updatednontilenodesptr,nupdatednontilenodesptr,
                          updatednontilenodesizeptr,inontilenodeoutarcptr,
                          totarclenptr,tag);
      }else{

        /* only one primary arc; just delete other secondary arc */
        /* find existing secondary arc (must be in this tile) */
        /* swap direction of existing secondary arc if necessary */
        arcnum=0;
        while(TRUE){
          if(scndryarcs(tilenum,arcnum).from==primarytail
             && scndryarcs(tilenum,arcnum).to==primaryhead){
            break;
          }else if(scndryarcs(tilenum,arcnum).from==primaryhead
                   && scndryarcs(tilenum,arcnum).to==primarytail){
            scndryarcs(tilenum,arcnum).from=primarytail;
            scndryarcs(tilenum,arcnum).to=primaryhead;
            break;
          }
          arcnum++;
        }

        /* assign cost of this secondary arc to existing secondary arc */
        scndrycosts(tilenum,arcnum)=scndrycostarr;

        /* update direction data in secondary arc structure */
        if(primarytail->col==primaryhead->col+1){
          scndryarcs(tilenum,arcnum).fromdir=RIGHT;
        }else if(primarytail->row==primaryhead->row+1){
          scndryarcs(tilenum,arcnum).fromdir=DOWN;
        }else if(primarytail->col==primaryhead->col-1){
          scndryarcs(tilenum,arcnum).fromdir=LEFT;
        }else{
          scndryarcs(tilenum,arcnum).fromdir=UP;
        }
      }

      /* we're done */
      return(0);
    }
  }

  /* set up secondary arc datastructures */
  nnewarcs=++(*nnewarcsptr);
  if(nnewarcs > SHRT_MAX){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Exceeded maximum number of secondary arcs. Decrease "
            "TILECOSTTHRESH and/or increase MINREGIONSIZE");
  }
  if(nnewarcs>scndryarcs.cols()){
    auto nnewcols=std::max(nnewarcs,2*scndryarcs.cols());
    scndryarcs.conservativeResize(Eigen::NoChange,nnewcols);
  }
  newarc=&scndryarcs(tilenum,nnewarcs-1);
  newarc->arcrow=tilenum;
  newarc->arccol=nnewarcs-1;
  if(nnewarcs>scndrycosts.cols()){
    auto nnewcols=std::max(nnewarcs,2*scndrycosts.cols());
    scndrycosts.conservativeResize(Eigen::NoChange,nnewcols);
  }
  scndrycosts(tilenum,nnewarcs-1)=scndrycostarr;

  /* update secondary node data */
  /* store primary nodes in nodesuppT neighbornodes[] arrays since */
  /* secondary node addresses change in ReAlloc() calls in TraceRegions() */
  supptail=&nodesupp(scndrytail->row,scndrytail->col);
  supphead=&nodesupp(scndryhead->row,scndryhead->col);
  supptail->noutarcs++;
  supptail->neighbornodes.conservativeResize(supptail->noutarcs);
  supptail->neighbornodes[supptail->noutarcs-1]=primaryhead;
  primarytail->level=scndrytail->row;
  primarytail->incost=scndrytail->col;
  supptail->outarcs.conservativeResize(supptail->noutarcs);
  supptail->outarcs[supptail->noutarcs-1]=NULL;
  supphead->noutarcs++;
  supphead->neighbornodes.conservativeResize(supphead->noutarcs);
  supphead->neighbornodes[supphead->noutarcs-1]=primarytail;
  primaryhead->level=scndryhead->row;
  primaryhead->incost=scndryhead->col;
  supphead->outarcs.conservativeResize(supphead->noutarcs);
  supphead->outarcs[supphead->noutarcs-1]=NULL;

  /* keep track of updated secondary nodes that were not in this tile */
  if(scndrytail->row!=tilenum){
    if(++(*nupdatednontilenodesptr)==(*updatednontilenodesizeptr)){
      (*updatednontilenodesizeptr)+=INITARRSIZE;
      updatednontilenodesptr->conservativeResize(*updatednontilenodesizeptr);
      inontilenodeoutarcptr->conservativeResize(*updatednontilenodesizeptr);
    }
    (*updatednontilenodesptr)[*nupdatednontilenodesptr-1]=scndrytail;
    (*inontilenodeoutarcptr)[*nupdatednontilenodesptr-1]=supptail->noutarcs-1;
  }
  if(scndryhead->row!=tilenum){
    if(++(*nupdatednontilenodesptr)==(*updatednontilenodesizeptr)){
      (*updatednontilenodesizeptr)+=INITARRSIZE;
      updatednontilenodesptr->conservativeResize(*updatednontilenodesizeptr);
      inontilenodeoutarcptr->conservativeResize(*updatednontilenodesizeptr);
    }
    (*updatednontilenodesptr)[*nupdatednontilenodesptr-1]=scndryhead;
    (*inontilenodeoutarcptr)[*nupdatednontilenodesptr-1]=supphead->noutarcs-1;
  }

  /* set up node data in secondary arc structure */
  newarc->from=primarytail;
  newarc->to=primaryhead;
  
  /* set up direction data in secondary arc structure */
  tempnode=primaryhead->pred;
  if(tempnode->col==primaryhead->col+1){
    newarc->fromdir=RIGHT;
  }else if(tempnode->row==primaryhead->row+1){
    newarc->fromdir=DOWN;
  }else if(tempnode->col==primaryhead->col-1){
    newarc->fromdir=LEFT;
  }else{
    newarc->fromdir=UP;
  }

  /* add number of primary arcs in secondary arc to counter */
  (*totarclenptr)+=arclen;

  /* done */
  return(0);

}


/* function: FindScndryNode()
 * --------------------------
 */
static
nodeT *FindScndryNode(Array2D<nodeT>& scndrynodes, Array2D<nodesuppT>& nodesupp,
                      long tilenum, long primaryrow, long primarycol){

  long nodenum;

  /* set temporary variables */
  auto nodesuppptr=nodesupp.row(tilenum);

  /* loop over all nodes in the tile until we find a match */
  nodenum=0;
  while(nodesuppptr[nodenum].row!=primaryrow
        || nodesuppptr[nodenum].col!=primarycol){
    nodenum++;
  }
  return(&scndrynodes(tilenum,nodenum));
}


/* function: IntegrateSecondaryFlows()
 * -----------------------------------
 */
static
int IntegrateSecondaryFlows(long linelen, long nlines, Array2D<nodeT>& /*scndrynodes*/,
                            Array2D<nodesuppT>& nodesupp, Array2D<scndryarcT>& scndryarcs,
                            Array1D<int>& nscndryarcs, Array2D<short>& scndryflows,
                            Array2D<short>& bulkoffsets, outfileT *outfiles,
                            paramT *params){

  FILE *outfp;
  long row, col, colstart, nrow, ncol, nnrow, nncol, maxcol;
  long readtilelinelen, readtilenlines, nextcoloffset, nextrowoffset;
  long tilerow, tilecol, ntilerow, ntilecol, rowovrlp, colovrlp;
  long ni, nj, tilenum;
  double tileoffset;
  char realoutfile[MAXSTRLEN]={}, readfile[MAXSTRLEN]={};
  char path[MAXSTRLEN]={}, basename[MAXSTRLEN]={};
  signed char writeerror;
  tileparamT readtileparams[1]={};
  outfileT readtileoutfiles[1]={};

  /* set up */
  auto info=pyre::journal::info_t("isce3.unwrap.snaphu");
  info << pyre::journal::at(__HERE__)
       << "Integrating secondary flows"
       << pyre::journal::endl;
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  rowovrlp=params->rowovrlp;
  colovrlp=params->colovrlp;
  ni=ceil((nlines+(ntilerow-1)*rowovrlp)/(double )ntilerow);
  nj=ceil((linelen+(ntilecol-1)*colovrlp)/(double )ntilecol);
  nextcoloffset=0;
  writeerror=FALSE;
  nrow=0;

  /* get memory */
  auto regions=Array2D<short>(ni,nj);
  auto tileflows=MakeRowColArray2D<short>(ni+2,nj+2);
  auto tileunwphase=Array2D<float>(ni,nj);
  auto tilemag=Array2D<float>(ni,nj);
  auto unwphase=Array2D<float>(ni,linelen);
  auto mag=Array2D<float>(ni,linelen);
  auto outline=Array1D<float>(2*linelen);

  /* flip sign of bulk offsets if flip flag is set */
  /* do this and flip flow signs instead of flipping phase signs */
  if(params->flipphasesign){
    for(row=0;row<ntilerow;row++){
      for(col=0;col<ntilecol;col++){
        bulkoffsets(row,col)*=-1;
      }
    }
  }

  /* open output file */
  outfp=OpenOutputFile(outfiles->outfile,realoutfile);

  /* process each tile row */
  for(tilerow=0;tilerow<ntilerow;tilerow++){
    
    /* process each tile column, place into unwrapped tile row array */
    nextrowoffset=0;
    for(tilecol=0;tilecol<ntilecol;tilecol++){

      /* use SetupTile() to set filenames; tile params overwritten below */
      SetupTile(nlines,linelen,params,readtileparams,outfiles,
                readtileoutfiles,tilerow,tilecol);
      colstart=readtileparams->firstcol;
      readtilenlines=readtileparams->nrow;
      readtilelinelen=readtileparams->ncol;

      /* set tile read parameters */
      SetTileReadParams(readtileparams,readtilenlines,readtilelinelen,
                        tilerow,tilecol,nlines,linelen,params);
      colstart+=readtileparams->firstcol;
      nrow=readtileparams->nrow;
      ncol=readtileparams->ncol;
      nnrow=nrow+1;
      nncol=ncol+1;

      /* read unwrapped phase */
      /* phase sign not flipped for positive baseline */
      /* since flow will be flipped if necessary */
      if(TMPTILEOUTFORMAT==ALT_LINE_DATA){
        ReadAltLineFile(&tilemag,&tileunwphase,readtileoutfiles->outfile,
                        readtilelinelen,readtilenlines,readtileparams);
      }else if(TMPTILEOUTFORMAT==FLOAT_DATA){
        Read2DArray(&tileunwphase,readtileoutfiles->outfile,
                    readtilelinelen,readtilenlines,readtileparams,
                    sizeof(float *),sizeof(float));
      }
        
      /* read regions */
      ParseFilename(outfiles->outfile,path,basename);
      auto tempstring=std::string(params->tiledir)+"/"+TMPTILEROOT+basename+"_"
        +std::to_string(tilerow)+"_"+std::to_string(tilecol)+"."
          +std::to_string(readtilelinelen)+REGIONSUFFIX;
      StrNCopy(readfile,tempstring.c_str(),MAXSTRLEN);
      Read2DArray(&regions,readfile,readtilelinelen,readtilenlines,
                  readtileparams,sizeof(short *),sizeof(short));

      /* remove temporary files unless told so save them */
      if(params->rmtmptile){
        unlink(readtileoutfiles->outfile);
        unlink(readfile);
      }

      /* zero out primary flow array */
      for(row=0;row<2*nrow+1;row++){
        if(row<nrow){
          maxcol=ncol;
        }else{
          maxcol=ncol+1;
        }
        for(col=0;col<maxcol;col++){
          tileflows(row,col)=0;
        }
      }

      /* loop over each secondary arc in this tile and parse flows */
      /* if flip flag set, flow derived from flipped phase array */
      /* flip flow for integration in ParseSecondaryFlows() */
      tilenum=tilerow*ntilecol+tilecol;
      ParseSecondaryFlows(tilenum,nscndryarcs,tileflows,regions,scndryflows,
                          nodesupp,scndryarcs,nrow,ncol,ntilerow,ntilecol,
                          params);

      /* place tile mag, adjusted unwrapped phase into output arrays */
      mag(0,colstart)=tilemag(0,0);
      if(tilecol==0){
        tileoffset=TWOPI*nextcoloffset;
      }else{
        tileoffset=TWOPI*nextrowoffset;
      }
      unwphase(0,colstart)=tileunwphase(0,0)+tileoffset;
      for(col=1;col<ncol;col++){
        mag(0,colstart+col)=tilemag(0,col);
        unwphase(0,colstart+col)
          =(float )((double )unwphase(0,colstart+col-1)
                    +(double )tileunwphase(0,col)
                    -(double )tileunwphase(0,col-1)
                    +tileflows(nnrow,col)*TWOPI);
      }
      if(tilecol!=ntilecol-1){
        nextrowoffset=(LRound((unwphase(0,colstart+ncol-1)
                               -tileunwphase(0,ncol-1))/TWOPI)
                       +tileflows(nnrow,nncol-1)
                       +bulkoffsets(tilerow,tilecol)
                       -bulkoffsets(tilerow,tilecol+1));
      }
      for(row=1;row<nrow;row++){
        for(col=0;col<ncol;col++){
          mag(row,colstart+col)=tilemag(row,col);
          unwphase(row,colstart+col)
            =(float )((double )unwphase(row-1,colstart+col)
                      +(double )tileunwphase(row,col)
                      -(double )tileunwphase(row-1,col)
                      -tileflows(row,col)*TWOPI);
        }
      }
      if(tilecol==0 && tilerow!=ntilerow-1){
        nextcoloffset=(LRound((unwphase(nrow-1,colstart)
                              -tileunwphase(nrow-1,0))/TWOPI)
                       -tileflows(nnrow-1,0)
                       +bulkoffsets(tilerow,tilecol)
                       -bulkoffsets(tilerow+1,tilecol));
      }

    } /* end loop over tile columns */

    /* write out tile row */
    for(row=0;row<nrow;row++){
      if(outfiles->outfileformat==ALT_LINE_DATA){
        if(fwrite(mag.row(row).data(),sizeof(float),linelen,outfp)!=linelen
           || fwrite(unwphase.row(row).data(),sizeof(float),linelen,outfp)!=linelen){
          writeerror=TRUE;
          break;
        }
      }else if(outfiles->outfileformat==ALT_SAMPLE_DATA){
        for(col=0;col<linelen;col++){
          outline[2*col]=mag(row,col);
          outline[2*col+1]=unwphase(row,col);
        }
        if(fwrite(outline.data(),sizeof(float),2*linelen,outfp)!=2*linelen){
          writeerror=TRUE;
          break;
        }
      }else{
        if(fwrite(unwphase.row(row).data(),sizeof(float),linelen,outfp)!=linelen){
          writeerror=TRUE;
          break;
        }
      }
    }
    if(writeerror){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while writing to file " + std::string(realoutfile) +
              " (device full?)");
    }

  } /* end loop over tile rows */


  /* close output file, free memory */
  info << pyre::journal::at(__HERE__)
       << "Integrating secondary flows"
       << "Output written to file " << realoutfile
       << pyre::journal::endl;
  if(fclose(outfp)){
    fflush(NULL);
    auto warnings=pyre::journal::warning_t("isce3.unwrap.snaphu");
    warnings << pyre::journal::at(__HERE__)
             << "WARNING: problem closing file " << realoutfile << " (disk full?)"
             << pyre::journal::endl;
  }
  return(0);

}


/* function: ParseSecondaryFlows()
 * -------------------------------
 */
static
int ParseSecondaryFlows(long tilenum, Array1D<int>& nscndryarcs, Array2D<short>& tileflows,
                         Array2D<short>& regions, Array2D<short>& scndryflows,
                         Array2D<nodesuppT>& nodesupp, Array2D<scndryarcT>& scndryarcs,
                         long nrow, long ncol, long /*ntilerow*/, long ntilecol,
                         paramT *params){

  nodeT *scndryfrom, *scndryto;
  long arcnum, nnrow, nncol, nflow, primaryfromrow, primaryfromcol;
  long prevrow, prevcol, thisrow, thiscol, nextrow, nextcol;
  signed char phaseflipsign;


  /* see if we need to flip sign of flow because of positive topo baseline */
  if(params->flipphasesign){
    phaseflipsign=-1;
  }else{
    phaseflipsign=1;
  }

  /* loop over all arcs in tile */
  for(arcnum=0;arcnum<nscndryarcs[tilenum];arcnum++){

    /* do nothing if prev arc has no secondary flow */
    nflow=phaseflipsign*scndryflows(tilenum,arcnum);
    if(nflow){

      /* get arc info */
      nnrow=nrow+1;
      nncol=ncol+1;
      scndryfrom=scndryarcs(tilenum,arcnum).from;
      scndryto=scndryarcs(tilenum,arcnum).to;
      if(scndryfrom->row==tilenum){
        primaryfromrow=nodesupp(scndryfrom->row,scndryfrom->col).row;
        primaryfromcol=nodesupp(scndryfrom->row,scndryfrom->col).col;
      }else if(scndryfrom->row==tilenum-ntilecol){
        primaryfromrow=0;
        primaryfromcol=nodesupp(scndryfrom->row,scndryfrom->col).col;
      }else if(scndryfrom->row==tilenum-1){
        primaryfromrow=nodesupp(scndryfrom->row,scndryfrom->col).row;
        primaryfromcol=0;
      }else{
        primaryfromrow=0;
        primaryfromcol=0;
      }
      if(scndryto->row==tilenum){
        thisrow=nodesupp(scndryto->row,scndryto->col).row;
        thiscol=nodesupp(scndryto->row,scndryto->col).col;
      }else if(scndryto->row==tilenum-ntilecol){
        thisrow=0;
        thiscol=nodesupp(scndryto->row,scndryto->col).col;
      }else if(scndryto->row==tilenum-1){
        thisrow=nodesupp(scndryto->row,scndryto->col).row;
        thiscol=0;
      }else{
        thisrow=0;
        thiscol=0;
      }

      /* set initial direction out of secondary arc head */
      switch(scndryarcs(tilenum,arcnum).fromdir){
      case RIGHT:
        nextrow=thisrow;
        nextcol=thiscol+1;
        tileflows(thisrow,thiscol)-=nflow;
        break;
      case DOWN:
        nextrow=thisrow+1;
        nextcol=thiscol;
        tileflows(nnrow+thisrow,thiscol)-=nflow;
        break;
      case LEFT:
        nextrow=thisrow;
        nextcol=thiscol-1;
        tileflows(thisrow,thiscol-1)+=nflow;
        break;
      default:
        nextrow=thisrow-1;
        nextcol=thiscol;
        tileflows(nnrow+thisrow-1,thiscol)+=nflow;
        break;
      }

      /* use region data to trace path between secondary from, to */
      while(!(nextrow==primaryfromrow && nextcol==primaryfromcol)){

        /* move to next node */
        prevrow=thisrow;
        prevcol=thiscol;
        thisrow=nextrow;
        thiscol=nextcol;
    
        /* check rightward arc */
        if(thiscol!=nncol-1){
          if(thisrow==0 || thisrow==nnrow-1
             || regions(thisrow-1,thiscol)!=regions(thisrow,thiscol)){
            if(!(thisrow==prevrow && thiscol+1==prevcol)){
              tileflows(thisrow,thiscol)-=nflow;
              nextcol++;
            }
          }
        }

        /* check downward arc */
        if(thisrow!=nnrow-1){
          if(thiscol==0 || thiscol==nncol-1
             || regions(thisrow,thiscol)!=regions(thisrow,thiscol-1)){
            if(!(thisrow+1==prevrow && thiscol==prevcol)){
              tileflows(nnrow+thisrow,thiscol)-=nflow;
              nextrow++;
            }
          }
        }
    
        /* check leftward arc */
        if(thiscol!=0){
          if(thisrow==0 || thisrow==nnrow-1
             || regions(thisrow,thiscol-1)!=regions(thisrow-1,thiscol-1)){
            if(!(thisrow==prevrow && thiscol-1==prevcol)){
              tileflows(thisrow,thiscol-1)+=nflow;
              nextcol--;
            }
          }
        }

        /* check upward arc */
        if(thisrow!=0){
          if(thiscol==0 || thiscol==nncol-1
             || regions(thisrow-1,thiscol-1)!=regions(thisrow-1,thiscol)){
            if(!(thisrow-1==prevrow && thiscol==prevcol)){
              tileflows(nnrow+thisrow-1,thiscol)+=nflow;
              nextrow--;
            }
          }
        }
      }   
    }
  }
  return(0);
}


/* function: AssembleTileConnComps()
 * ---------------------------------
 * Assemble conntected components per tile.
 */
static
int AssembleTileConnComps(long linelen, long nlines,
                          outfileT *outfiles, paramT *params){

  int ipass;
  long k;
  long row, col, colstart, nrow, ncol;
  long readtilelinelen, readtilenlines;
  long tilerow, tilecol, ntilerow, ntilecol, rowovrlp, colovrlp;
  long ni, nj, tilenum;
  unsigned int iconncomp, iconncompmax;
  long ntileconncomp, nconncomp;
  long ntileconncompmem, nconncompmem, nmemold;
  char realoutfile[MAXSTRLEN]={};
  signed char writeerror;
  tileparamT readtileparams[1]={};
  outfileT readtileoutfiles[1]={};
  FILE *outfp;

  /* set up */
  auto info=pyre::journal::info_t("isce3.unwrap.snaphu");
  info << pyre::journal::at(__HERE__)
       << "Assembling tile connected components"
       << pyre::journal::endl;
  ntilerow=params->ntilerow;
  ntilecol=params->ntilecol;
  rowovrlp=params->rowovrlp;
  colovrlp=params->colovrlp;
  ni=ceil((nlines+(ntilerow-1)*rowovrlp)/(double )ntilerow);
  nj=ceil((linelen+(ntilecol-1)*colovrlp)/(double )ntilecol);
  writeerror=FALSE;
  nrow=0;
  nconncomp=0;
  nconncompmem=0;
  ntileconncompmem=0;
  iconncompmax=0;

  Array1D<conncompsizeT> tileconncompsizes, conncompsizes;
  Array1D<unsigned> tilemapping;

  /* get memory */
  auto tileconncomps=Array2D<unsigned>(ni,nj);
  auto tilerowconncomps=Array2D<unsigned>(ni,linelen);
  auto ucharbuf=Array2D<unsigned char>(ni,nj);
  auto ucharoutbuf=Array1D<unsigned char>(linelen);

  /* open output file */
  outfp=OpenOutputFile(outfiles->conncompfile,realoutfile);

  /* do two passes looping over all tiles */
  for(ipass=0;ipass<2;ipass++){

    /* process each tile row */
    for(tilerow=0;tilerow<ntilerow;tilerow++){
    
      /* process each tile column */
      for(tilecol=0;tilecol<ntilecol;tilecol++){

        /* use SetupTile() to set filenames; tile params overwritten below */
        SetupTile(nlines,linelen,params,readtileparams,outfiles,
                  readtileoutfiles,tilerow,tilecol);
        colstart=readtileparams->firstcol;
        readtilenlines=readtileparams->nrow;
        readtilelinelen=readtileparams->ncol;

        /* set tile read parameters */
        SetTileReadParams(readtileparams,readtilenlines,readtilelinelen,
                          tilerow,tilecol,nlines,linelen,params);
        colstart+=readtileparams->firstcol;
        nrow=readtileparams->nrow;
        ncol=readtileparams->ncol;

        /* set tile number */
        tilenum=tilerow*ntilecol+tilecol;

        /* read connected components for tile */
        if(params->conncompouttype==CONNCOMPOUTTYPEUCHAR){
          Read2DArray(&ucharbuf,readtileoutfiles->conncompfile,
                      readtilelinelen,readtilenlines,readtileparams,
                      sizeof(unsigned char *),sizeof(unsigned char));
          for(row=0;row<nrow;row++){
            for(col=0;col<ncol;col++){
              tileconncomps(row,col)=(unsigned int )ucharbuf(row,col);
            }
          }
        }else{
          Read2DArray(&tileconncomps,readtileoutfiles->conncompfile,
                      readtilelinelen,readtilenlines,readtileparams,
                      sizeof(unsigned int *),sizeof(unsigned int));
        }
      
        /* see which pass we are in */
        if(ipass==0){

          /* first pass */
          
          /* initialize tileconncomps array for this tile */
          ntileconncomp=0;
          for(k=0;k<ntileconncompmem;k++){
            tileconncompsizes[k].tilenum=tilenum;
            tileconncompsizes[k].icomptile=k+1;
            tileconncompsizes[k].icompfull=0;
            tileconncompsizes[k].npix=0;
          }
          
          /* loop over kept pixels and count pixels in each conncomp */
          for(row=0;row<nrow;row++){
            for(col=0;col<ncol;col++){

              /* see if have connected component (conncomp number not zero) */
              iconncomp=tileconncomps(row,col);
              if(iconncomp>0){

                /* get more memory for tile conncompsizeT array if needed */
                while(iconncomp>ntileconncompmem){
                  nmemold=ntileconncompmem;
                  ntileconncompmem+=CONNCOMPMEMINCR;
                  tileconncompsizes.conservativeResize(ntileconncompmem);
                  for(k=nmemold;k<ntileconncompmem;k++){
                    tileconncompsizes[k].tilenum=tilenum;
                    tileconncompsizes[k].icomptile=k+1;
                    tileconncompsizes[k].icompfull=0;
                    tileconncompsizes[k].npix=0;
                  }
                }

                /* count number of connected components in tile */
                if(tileconncompsizes[iconncomp-1].npix==0){
                  tileconncompsizes[iconncomp-1].icomptile=iconncomp;
                  ntileconncomp++;
                }

                /* count number of pixels in this connected component */
                tileconncompsizes[iconncomp-1].npix++;

                /* keep max number of connected components in any tile */
                if(iconncomp>iconncompmax){
                  iconncompmax=iconncomp;
                }
                
              }
            }
          }

          /* get more memory for full set of connected components sizes */
          nmemold=nconncompmem;
          nconncompmem+=ntileconncomp;
          conncompsizes.conservativeResize(nconncompmem);

          /* store conncomp sizes from tile in full list */
          for(k=0;k<ntileconncompmem;k++){
            if(tileconncompsizes[k].npix>0){
              conncompsizes[nconncomp].tilenum=tileconncompsizes[k].tilenum;
              conncompsizes[nconncomp].icomptile=tileconncompsizes[k].icomptile;
              conncompsizes[nconncomp].icompfull=0;
              conncompsizes[nconncomp].npix=tileconncompsizes[k].npix;
              nconncomp++;
            }
          }
            
        }else{

          /* second pass */
          
          /* build lookup table for tile mapping for this tile */
          /* lookup table index is conncomp number minus one */
          for(k=0;k<iconncompmax;k++){
            tilemapping[k]=0;
          }
          for(k=0;k<nconncomp;k++){
            if(conncompsizes[k].tilenum==tilenum){
              iconncomp=conncompsizes[k].icomptile;
              tilemapping[iconncomp-1]=conncompsizes[k].icompfull;
            }
          }

          /* assign final conncomp number to output */
          for(row=0;row<nrow;row++){
            for(col=0;col<ncol;col++){
              iconncomp=tileconncomps(row,col);
              if(iconncomp>0){
                tilerowconncomps(row,colstart+col)=tilemapping[iconncomp-1];
              }else{
                tilerowconncomps(row,colstart+col)=0;
              }
            }
          }
          
          /* remove temporary files unless told so save them */
          if(params->rmtmptile){
            unlink(readtileoutfiles->conncompfile);
          }

        }

      } /* end loop over tile columns */

      /* write out tile row at end of second pass */
      if(ipass>0){
        for(row=0;row<nrow;row++){
          if(params->conncompouttype==CONNCOMPOUTTYPEUCHAR){
            for(k=0;k<linelen;k++){
              ucharoutbuf[k]=(unsigned char)tilerowconncomps(row,k);
            }
            if(fwrite(ucharoutbuf.data(),sizeof(unsigned char),linelen,outfp)
               !=linelen){
              writeerror=TRUE;
              break;
            }
          }else{
            if(fwrite(tilerowconncomps.row(row).data(),
                      sizeof(unsigned int),linelen,outfp)!=linelen){
              writeerror=TRUE;
              break;
            }
          }
        }
        if(writeerror){
          fflush(NULL);
          throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                  "Error while writing to file " + std::string(realoutfile) +
                  " (device full?)");
        }
      }

    } /* end loop over tile rows */

    /* at end of first pass, set up tile size array for next pass */
    if(ipass==0){

      /* sort tile size array into descending order */
      qsort(conncompsizes.data(),nconncomp,sizeof(conncompsizeT),
            ConnCompSizeNPixCompare);

      /* keep no more than max number of connected components */
      if(nconncomp>params->maxncomps){
        nconncomp=params->maxncomps;
      }
      
      /* assign tile mappings */
      for(k=0;k<nconncomp;k++){
        conncompsizes[k].icompfull=k+1;
      }

      /* get memory for tile mapping lookup table */
      tilemapping.conservativeResize(iconncompmax);

    }

  } /* end loop over passes of tile reading */

  /* close output file */
  info << pyre::journal::at(__HERE__)
       << "Assembled connected components (" << nconncomp
       << ") output written to file " << realoutfile
       << pyre::journal::endl;
  if(fclose(outfp)){
    fflush(NULL);
    auto warnings=pyre::journal::warning_t("isce3.unwrap.snaphu");
    warnings << pyre::journal::at(__HERE__)
             << "WARNING: problem closing file " << realoutfile << " (disk full?)"
             << pyre::journal::endl;
  }

  /* done */
  return(0);

}


/* function: ConnCompSizeNPixCompare()
 * -----------------------------------
 * Compare npix member of conncompsizeT structures pointed to by
 * inputs for use with qsort() into descending order.
 */
static
int ConnCompSizeNPixCompare(const void *ptr1, const void *ptr2){
  return(((conncompsizeT *)ptr2)->npix-((conncompsizeT *)ptr1)->npix);
}


#define INSTANTIATE_TEMPLATES(T) \
  template int GrowRegions(Array2D<typename T::Cost>&, Array2D<short>&, long, long, \
                           Array2D<incrcostT>&, outfileT*, \
                           tileparamT*, paramT*, T); \
  template int GrowConnCompsMask(Array2D<typename T::Cost>&, Array2D<short>&, long, long, \
                                 Array2D<incrcostT>&, outfileT*, \
                                 paramT*, T); \
  template int AssembleTiles(outfileT*, paramT*, \
                             long, long, T);
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
