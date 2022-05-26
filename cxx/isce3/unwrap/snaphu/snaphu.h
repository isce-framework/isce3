/*************************************************************************

  snaphu header file
  Written by Curtis W. Chen
  Copyright 2002 Board of Trustees, Leland Stanford Jr. University
  Please see the supporting documentation for terms of use.
  No warranty.

*************************************************************************/

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include <string>

#include <Eigen/Core>
#include <pyre/journal.h>

#include <isce3/except/Error.h>

/**********************/
/* defined constants  */
/**********************/

#define PROGRAMNAME          "snaphu"
#define VERSION              "2.0.5"
#define BUGREPORTEMAIL       "snaphu@gmail.com"
#ifdef PI
#undef PI
#endif
#define PI                   3.14159265358979323846
#define TWOPI                6.28318530717958647692
#define SQRTHALF             0.70710678118654752440
#define MAXSTRLEN            512
#define MAXTMPSTRLEN         1024
#define MAXLINELEN           2048
#define TRUE                 1
#define FALSE                0
#define LARGESHORT           32000
#define LARGEINT             2000000000
#define LARGEFLOAT           1.0e35
#define VERYFAR              LARGEINT
#define GROUNDROW            -2
#define GROUNDCOL            -2
#define BOUNDARYROW          -4
#define BOUNDARYCOL          -4
#define MAXGROUPBASE         LARGEINT
#define ONTREE               -1
#define INBUCKET             -2
#define NOTINBUCKET          -3
#define PRUNED               -4
#define MASKED               -5
#define BOUNDARYPTR          -6
#define BOUNDARYCANDIDATE    -7
#define BOUNDARYLEVEL        LARGEINT
#define INTERIORLEVEL        (BOUNDARYLEVEL-1)
#define MINBOUNDARYSIZE      100
#define POSINCR              0
#define NEGINCR              1
#define NOCOSTSHELF          -LARGESHORT
#define MINSCALARCOST        1
#define INITARRSIZE          500
#define NEWNODEBAGSTEP       500
#define CANDIDATEBAGSTEP     500
#define NEGBUCKETFRACTION    1.0
#define POSBUCKETFRACTION    1.0
#define CLIPFACTOR           0.6666666667
#define NSOURCELISTMEMINCR   1024
#define NLISTMEMINCR         1024
#define DEF_OUTFILE          "snaphu.out"
#define DEF_SYSCONFFILE      ""     /* "/usr/local/snaphu/snaphu.conf" */
#define DEF_WEIGHTFILE       ""     /* "snaphu.weight" */
#define DEF_AMPFILE          ""     /* "snaphu.amp" */
#define DEF_AMPFILE2         ""     /* "snaphu.amp" */
#define DEF_MAGFILE          ""     /* "snaphu.mag" */
#define DEF_CORRFILE         ""     /* "snaphu.corr" */
#define DEF_ESTFILE          ""     /* "snaphu.est" */
#define DEF_COSTINFILE       ""
#define DEF_BYTEMASKFILE     ""
#define DEF_DOTILEMASKFILE   ""
#define DEF_INITFILE         ""
#define DEF_FLOWFILE         ""
#define DEF_EIFILE           ""
#define DEF_ROWCOSTFILE      ""
#define DEF_COLCOSTFILE      ""
#define DEF_MSTROWCOSTFILE   ""
#define DEF_MSTCOLCOSTFILE   ""
#define DEF_MSTCOSTSFILE     ""
#define DEF_CORRDUMPFILE     ""
#define DEF_RAWCORRDUMPFILE  ""
#define DEF_CONNCOMPFILE     ""
#define DEF_COSTOUTFILE      ""
#define DEF_LOGFILE          ""
#define MAXITERATION         5000
#define NEGSHORTRANGE        SHRT_MIN
#define POSSHORTRANGE        SHRT_MAX
#define MAXRES               SCHAR_MAX
#define MINRES               SCHAR_MIN
#define PROBCOSTP            (-99.999)
#define NULLFILE             "/dev/null"
#define DEF_INITONLY         FALSE
#define DEF_INITMETHOD       MSTINIT
#define DEF_UNWRAPPED        FALSE
#define DEF_REGROWCONNCOMPS  FALSE
#define DEF_EVAL             FALSE
#define DEF_WEIGHT           1
#define DEF_COSTMODE         TOPO
#define DEF_AMPLITUDE        TRUE
#define AUTOCALCSTATMAX      0
#define MAXNSHORTCYCLE       8192
#define USEMAXCYCLEFRACTION  (-123)
#define COMPLEX_DATA         1         /* file format */
#define FLOAT_DATA           2         /* file format */
#define ALT_LINE_DATA        3         /* file format */
#define ALT_SAMPLE_DATA      4         /* file format */
#define TILEINITFILEFORMAT   ALT_LINE_DATA
#define TILEINITFILEROOT     "snaphu_tileinit_"
#define ABNORMAL_EXIT        1         /* exit code */
#define NORMAL_EXIT          0         /* exit code */
#define DUMP_PATH            "/tmp/"   /* default location for writing dumps */
#define NARMS                8         /* number of arms for Despeckle() */
#define ARMLEN               5         /* length of arms for Despeckle() */
#define KEDGE                5         /* length of edge detection window */
#define ARCUBOUND            200       /* capacities for MCF solver */
#define MSTINIT              1         /* initialization method */
#define MCFINIT              2         /* initialization method */
#define BIGGESTDZRHOMAX      10000.0
#define SECONDSPERPIXEL      0.000001  /* for delay between thread creations */
#define MAXTHREADS           64
#define TMPTILEDIRROOT       "snaphu_tiles_"
#define TILEDIRMODE          511
#define TMPTILEROOT          "tmptile_"
#define TMPTILECOSTSUFFIX    "cost_"
#define TMPTILEOUTFORMAT     ALT_LINE_DATA
#define REGIONSUFFIX         "_regions"
#define LOGFILEROOT          "tmptilelog_"
#define RIGHT                1
#define DOWN                 2
#define LEFT                 3
#define UP                   4
#define TILEDPSICOLFACTOR    0.8
#define TILEOVRLPWARNTHRESH  400
#define ZEROCOSTARC          -LARGEINT
#define PINGPONG             2
#define SINGLEANTTRANSMIT    1
#define NOSTATCOSTS          0
#define TOPO                 1
#define DEFO                 2
#define SMOOTH               3
#define CONNCOMPOUTTYPEUCHAR 1
#define CONNCOMPOUTTYPEUINT  4


/* SAR and geometry parameter defaults */

#define DEF_ORBITRADIUS      7153000.0
#define DEF_ALTITUDE         0.0
#define DEF_EARTHRADIUS      6378000.0
#define DEF_BASELINE         150.0
#define DEF_BASELINEANGLE    (1.25*PI)
#define DEF_BPERP            0
#define DEF_TRANSMITMODE     PINGPONG
#define DEF_NLOOKSRANGE      1
#define DEF_NLOOKSAZ         5
#define DEF_NLOOKSOTHER      1
#define DEF_NCORRLOOKS       23.8
#define DEF_NCORRLOOKSRANGE  3  
#define DEF_NCORRLOOKSAZ     15
#define DEF_NEARRANGE        831000.0
#define DEF_DR               8.0
#define DEF_DA               20.0 
#define DEF_RANGERES         10.0
#define DEF_AZRES            6.0
#define DEF_LAMBDA           0.0565647


/* scattering model defaults */

#define DEF_KDS              0.02
#define DEF_SPECULAREXP      8.0
#define DEF_DZRCRITFACTOR    2.0
#define DEF_SHADOW           FALSE
#define DEF_DZEIMIN          -4.0
#define DEF_LAYWIDTH         16 
#define DEF_LAYMINEI         1.25
#define DEF_SLOPERATIOFACTOR 1.18
#define DEF_SIGSQEI          100.0


/* decorrelation model parameters */

#define DEF_DRHO             0.005
#define DEF_RHOSCONST1       1.3
#define DEF_RHOSCONST2       0.14
#define DEF_CSTD1            0.4
#define DEF_CSTD2            0.35
#define DEF_CSTD3            0.06
#define DEF_DEFAULTCORR      0.01
#define DEF_RHOMINFACTOR     1.3


/* pdf model parameters */

#define DEF_DZLAYPEAK        -2.0
#define DEF_AZDZFACTOR       0.99
#define DEF_DZEIFACTOR       4.0 
#define DEF_DZEIWEIGHT       0.5 
#define DEF_DZLAYFACTOR      1.0
#define DEF_LAYCONST         0.9
#define DEF_LAYFALLOFFCONST  2.0
#define DEF_SIGSQSHORTMIN    1
#define DEF_SIGSQLAYFACTOR   0.1


/* deformation mode parameters */

#define DEF_DEFOAZDZFACTOR   1.0
#define DEF_DEFOTHRESHFACTOR 1.2
#define DEF_DEFOMAX          1.2
#define DEF_SIGSQCORR        0.05
#define DEF_DEFOLAYCONST     0.9


/* algorithm parameters */

#define DEF_FLIPPHASESIGN    FALSE
#define DEF_ONETILEREOPT     FALSE
#define DEF_RMTILEINIT       TRUE
#define DEF_MAXFLOW          4
#define DEF_KROWEI           65
#define DEF_KCOLEI           257
#define DEF_KPARDPSI         7
#define DEF_KPERPDPSI        7
#define DEF_THRESHOLD        0.001
#define DEF_INITDZR          2048.0
#define DEF_INITDZSTEP       100.0
#define DEF_MAXCOST          1000.0
#define DEF_COSTSCALE        100.0 
#define DEF_COSTSCALEAMBIGHT 80.0 
#define DEF_DNOMINCANGLE     0.01
#define DEF_SRCROW           -1
#define DEF_SRCCOL           -1
#define DEF_P                PROBCOSTP
#define DEF_BIDIRLPN         TRUE
#define DEF_NSHORTCYCLE      200
#define DEF_MAXNEWNODECONST  0.0008
#define DEF_MAXCYCLEFRACTION 0.00001
#define DEF_NCONNNODEMIN     0
#define DEF_MAXNFLOWCYCLES   USEMAXCYCLEFRACTION
#define DEF_INITMAXFLOW      9999
#define INITMAXCOSTINCR      200
#define NOSTATINITMAXFLOW    15
#define DEF_ARCMAXFLOWCONST  3
#define DEF_DUMPALL          FALSE
#define DUMP_INITFILE        "snaphu.init"
#define DUMP_FLOWFILE        "snaphu.flow"
#define DUMP_EIFILE          "snaphu.ei"
#define DUMP_ROWCOSTFILE     "snaphu.rowcost"
#define DUMP_COLCOSTFILE     "snaphu.colcost"
#define DUMP_MSTROWCOSTFILE  "snaphu.mstrowcost"
#define DUMP_MSTCOLCOSTFILE  "snaphu.mstcolcost"
#define DUMP_MSTCOSTSFILE    "snaphu.mstcosts"
#define DUMP_CORRDUMPFILE    "snaphu.corr"
#define DUMP_RAWCORRDUMPFILE "snaphu.rawcorr"
#define INCRCOSTFILEPOS      "snaphu.incrcostpos"
#define INCRCOSTFILENEG      "snaphu.incrcostneg"
#define DEF_NMAJORPRUNE      LARGEINT
#define DEF_PRUNECOSTTHRESH  LARGEINT
#define DEF_EDGEMASKTOP      0
#define DEF_EDGEMASKBOT      0
#define DEF_EDGEMASKLEFT     0
#define DEF_EDGEMASKRIGHT    0
#define CONNCOMPMEMINCR      1024


/* default tile parameters */

#define DEF_NTILEROW         1
#define DEF_NTILECOL         1
#define DEF_ROWOVRLP         0
#define DEF_COLOVRLP         0
#define DEF_PIECEFIRSTROW    1
#define DEF_PIECEFIRSTCOL    1
#define DEF_PIECENROW        0
#define DEF_PIECENCOL        0
#define DEF_TILECOSTTHRESH   500
#define DEF_MINREGIONSIZE    100
#define DEF_NTHREADS         1
#define DEF_SCNDRYARCFLOWMAX 8
#define DEF_TILEEDGEWEIGHT   2.5
#define DEF_TILEDIR          ""
#define DEF_ASSEMBLEONLY     FALSE
#define DEF_RMTMPTILE        TRUE


/* default connected component parameters */
#define DEF_MINCONNCOMPFRAC  0.01
#define DEF_CONNCOMPTHRESH   300
#define DEF_MAXNCOMPS        32
#define DEF_CONNCOMPOUTTYPE  CONNCOMPOUTTYPEUCHAR


/* default file formats */

#define DEF_INFILEFORMAT              COMPLEX_DATA
#define DEF_UNWRAPPEDINFILEFORMAT     ALT_LINE_DATA
#define DEF_MAGFILEFORMAT             FLOAT_DATA
#define DEF_OUTFILEFORMAT             ALT_LINE_DATA
#define DEF_CORRFILEFORMAT            ALT_LINE_DATA
#define DEF_ESTFILEFORMAT             ALT_LINE_DATA
#define DEF_AMPFILEFORMAT             ALT_SAMPLE_DATA

/* command-line usage help strings */

#define OPTIONSHELPFULL\
 "usage:  snaphu [options] infile linelength [options]\n"\
 "options:\n"\
 "  -t              use topography mode costs (default)\n"\
 "  -d              use deformation mode costs\n"\
 "  -s              use smooth-solution mode costs\n"\
 "  -C <confstr>    parse argument string as config line as from conf file\n"\
 "  -f <filename>   read configuration parameters from file\n"\
 "  -o <filename>   write output to file\n"\
 "  -a <filename>   read amplitude data from file\n"\
 "  -A <filename>   read power data from file\n"\
 "  -m <filename>   read interferogram magnitude data from file\n"\
 "  -M <filename>   read byte mask data from file\n"\
 "  -c <filename>   read correlation data from file\n"\
 "  -e <filename>   read coarse unwrapped-phase estimate from file\n"\
 "  -w <filename>   read scalar weights from file\n"\
 "  -b <decimal>    perpendicular baseline (meters, topo mode only)\n"\
 "  -p <decimal>    Lp-norm parameter p\n"\
 "  -i              do initialization and exit\n"\
 "  -n              do not use statistical costs (with -p or -i)\n"\
 "  -u              infile is already unwrapped; initialization not needed\n"\
 "  -q              quantify cost of unwrapped input file then exit\n"\
 "  -g <filename>   grow connected components mask and write to file\n"\
 "  -G <filename>   grow connected components mask for unwrapped input\n"\
 "  -S              single-tile reoptimization after multi-tile init\n"\
 "  -k              keep temporary tile outputs\n"\
 "  -l <filename>   log runtime parameters to file\n"\
 "  -v              give verbose output\n"\
 "  --mst           use MST algorithm for initialization (default)\n"\
 "  --mcf           use MCF algorithm for initialization\n"\
 "  --aa <filename1> <filename2>    read amplitude from next two files\n"\
 "  --AA <filename1> <filename2>    read power from next two files\n"\
 "  --costinfile <filename>         read statistical costs from file\n"\
 "  --costoutfile <filename>        write statistical costs to file\n"\
 "  --tile <nrow> <ncol> <rowovrlp> <colovrlp>  unwrap as nrow x ncol tiles\n"\
 "  --nproc <integer>               number of processors used in tile mode\n"\
 "  --tiledir <dirname>             use specified directory for tiles\n"\
 "  --assemble                      assemble unwrapped tiles in tiledir\n"\
 "  --piece <firstrow> <firstcol> <nrow> <ncol>  unwrap subset of image\n" \
 "  --debug, --dumpall              dump all intermediate data arrays\n"\
 "  --copyright, --info             print copyright and bug report info\n"\
 "  -h, --help                      print this help text\n"\
 "\n"

#define OPTIONSHELPBRIEF\
 "usage:  snaphu [options] infile linelength [options]\n"\
 "most common options:\n"\
 "  -t              use topography mode costs (default)\n"\
 "  -d              use deformation mode costs\n"\
 "  -s              use smooth-solution mode costs\n"\
 "  -C <confstr>    parse argument string as config line as from conf file\n"\
 "  -f <filename>   read configuration parameters from file\n"\
 "  -o <filename>   write output to file\n"\
 "  -a <filename>   read amplitude data from file\n"\
 "  -c <filename>   read correlation data from file\n"\
 "  -M <filename>   read byte mask data from file\n"\
 "  -b <decimal>    perpendicular baseline (meters)\n"\
 "  -i              do initialization and exit\n"\
 "  -S              single-tile reoptimization after multi-tile init\n"\
 "  -l <filename>   log runtime parameters to file\n"\
 "  -u              infile is already unwrapped; initialization not needed\n"\
 "  -v              give verbose output\n"\
 "  --mst           use MST algorithm for initialization (default)\n"\
 "  --mcf           use MCF algorithm for initialization\n"\
 "  --tile <nrow> <ncol> <rowovrlp> <colovrlp>  unwrap as nrow x ncol tiles\n"\
 "  --nproc <integer>               number of processors used in tile mode\n"\
 "\n"\
 "type snaphu -h for a complete list of options\n"\
 "\n"

#define COPYRIGHT\
 "Written by Curtis W. Chen\n"\
 "Copyright 2002,2017 Board of Trustees, Leland Stanford Jr. University\n"\
 "\n"\
 "Except as noted below, permission to use, copy, modify, and\n"\
 "distribute, this software and its documentation for any purpose is\n"\
 "hereby granted without fee, provided that the above copyright notice\n"\
 "appear in all copies and that both that copyright notice and this\n"\
 "permission notice appear in supporting documentation, and that the\n"\
 "name of the copyright holders be used in advertising or publicity\n"\
 "pertaining to distribution of the software with specific, written\n"\
 "prior permission, and that no fee is charged for further distribution\n"\
 "of this software, or any modifications thereof.  The copyright holder\n"\
 "makes no representations about the suitability of this software for\n"\
 "any purpose.  It is provided \"as is\" without express or implied\n"\
 "warranty.\n"\
 "\n"\
 "THE COPYRIGHT HOLDER DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS\n"\
 "SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND\n"\
 "FITNESS, IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY\n"\
 "SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER\n"\
 "RESULTING FROM LOSS OF USE, DATA, PROFITS, QPA OR GPA, WHETHER IN AN\n"\
 "ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT\n"\
 "OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.\n"\
 "\n"\
 "\n"\
 "Please send snaphu bug reports to " BUGREPORTEMAIL "\n"\
 "\n"


namespace isce3::unwrap {

/********************/
/* type definitions */
/********************/

/* 1-D dynamically-sized owning array */
template<typename T>
using Array1D = Eigen::Array<T, Eigen::Dynamic, 1>;

/* 2-D row-major dynamically-sized owning array */
template<typename T>
using Array2D = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/* node data structure */
typedef struct nodeST{
  int row=0,col=0;              /* row, col of this node */
  struct nodeST *next=nullptr;  /* ptr to next node in thread or bucket */
  struct nodeST *prev=nullptr;  /* ptr to previous node in thread or bucket */
  struct nodeST *pred=nullptr;  /* parent node in tree */
  unsigned int level=0;         /* tree level */
  int group=0;                  /* for marking label */
  int incost=0,outcost=0;       /* costs to, from root of tree */
}nodeT;


/* boundary neighbor structure */
typedef struct neighborST{
  nodeT *neighbor=nullptr;      /* neighbor node pointer */
  int arcrow=0;                 /* row of arc to neighbor */
  int arccol=0;                 /* col of arc to neighbor */
  int arcdir=0;                 /* direction of arc to neighbor */
}neighborT;


/* boundary data structure */
typedef struct boundaryST{
  nodeT node[1]={};             /* ground node pointed to by this boundary */
  Array1D<neighborT> neighborlist={}; /* list of neighbors of common boundary */
  Array1D<nodeT*> boundarylist={};  /* list of nodes covered by common boundary */
  long nneighbor=0;             /* number of neighbor nodes of boundary */
  long nboundary=0;             /* number of nodes covered by boundary */
}boundaryT;

  
/* arc cost data structure */
typedef struct costST{
  short offset=0;               /* offset of wrapped phase gradient from 0 */
  short sigsq=0;                /* variance due to decorrelation */
  short dzmax=0;                /* largest discontinuity on shelf */
  short laycost=0;              /* cost of layover discontinuity shelf */
}costT;


/* arc cost data structure for smooth costs */
typedef struct smoothcostST{
  short offset=0;               /* offset of wrapped phase gradient from 0 */
  short sigsq=0;                /* variance due to decorrelation */
}smoothcostT;


/* arc cost data structure for bidirectional scalar costs */
typedef struct bidircostST{
  short posweight=0;            /* weight for positive flows */
  short negweight=0;            /* weight for negative flows */
}bidircostT;


/* incremental cost data structure */
typedef struct incrcostST{
  short poscost=0;              /* cost for positive flow increment */
  short negcost=0;              /* cost for negative flow increment */
}incrcostT;


/* arc candidate data structure */
typedef struct candidateST{
  nodeT *from=nullptr, *to=nullptr; /* endpoints of candidate arc */
  long violation=0;             /* magnitude of arc violation */
  int arcrow=0,arccol=0;        /* indexes into arc arrays */
  signed char arcdir=0;         /* direction of arc (1=fwd, -1=rev) */
}candidateT;


/* bucket data structure */
typedef struct bucketST{
  long size=0;                  /* number of buckets in list */
  long curr=0;                  /* current bucket index */
  long maxind=0;                /* maximum bucket index */
  long minind=0;                /* smallest (possibly negative) bucket index */
  nodeT **bucket=nullptr;       /* array of first nodes in each bucket */
  Array1D<nodeT*> bucketbase={};  /* real base of bucket array */
  signed char wrapped=FALSE;    /* flag denoting wrapped circular buckets */
}bucketT;


/* secondary arc data structure */
typedef struct scndryarcST{
  int arcrow=0;                 /* row of arc in secondary network array */
  int arccol=0;                 /* col of arc in secondary network array */
  nodeT *from=nullptr;          /* secondary node at tail of arc */
  nodeT *to=nullptr;            /* secondary node at head of arc */
  signed char fromdir=0;        /* direction from which arc enters head */
}scndryarcT;


/* supplementary data structure for secondary nodes */
typedef struct nodesuppST{
  int row=0;                    /* row of node in primary network problem */
  int col=0;                    /* col of node in primary network problem */
  Array1D<nodeT*> neighbornodes={}; /* pointers to neighboring secondary nodes */
  Array1D<scndryarcT*> outarcs={};  /* pointers to secondary arcs to neighbors */
  int noutarcs=0;               /* number of arcs from this node */
}nodesuppT;


/* run-time parameter data structure */
typedef struct paramST{

  /* SAR and geometry parameters */
  double orbitradius=0.0;       /* radius of platform orbit (meters) */
  double altitude=0.0;          /* SAR altitude (meters) */
  double earthradius=0.0;       /* radius of earth (meters) */
  double bperp=0.0;             /* nominal perpendiuclar baseline (meters) */
  signed char transmitmode=0;   /* transmit mode (PINGPONG or SINGLEANTTRANSMIT) */
  double baseline=0.0;          /* baseline length (meters, always postive) */
  double baselineangle=0.0;     /* baseline angle above horizontal (rad) */
  long nlooksrange=0;           /* number of looks in range for input data */
  long nlooksaz=0;              /* number of looks in azimuth for input data */
  long nlooksother=0;           /* number of nonspatial looks for input data */
  double ncorrlooks=0.0;        /* number of independent looks in correlation est */
  long ncorrlooksrange=0;       /* number of looks in range for correlation */
  long ncorrlooksaz=0;          /* number of looks in azimuth for correlation */
  double nearrange=0.0;         /* slant range to near part of swath (meters) */
  double dr=0.0;                /* range bin spacing (meters) */
  double da=0.0;                /* azimuth bin spacing (meters) */
  double rangeres=0.0;          /* range resolution (meters) */
  double azres=0.0;             /* azimuth resolution (meters) */
  double lambda=0.0;            /* wavelength (meters) */

  /* scattering model parameters */
  double kds=0.0;               /* ratio of diffuse to specular scattering */
  double specularexp=0.0;       /* power specular scattering component */
  double dzrcritfactor=0.0;     /* fudge factor for linearizing scattering model */
  signed char shadow=0;         /* allow discontinuities from shadowing */
  double dzeimin=0.0;           /* lower limit for backslopes (if shadow = FALSE) */
  long laywidth=0;              /* width of window for summing layover brightness */
  double layminei=0.0;          /* threshold brightness for assuming layover */
  double sloperatiofactor=0.0;  /* fudge factor for linearized scattering slopes */
  double sigsqei=0.0;           /* variance (dz, meters) due to uncertainty in EI */

  /* decorrelation model parameters */
  double drho=0.0;              /* step size of correlation-slope lookup table */
  double rhosconst1=0.0,rhosconst2=0.0; /* for calculating rho0 in biased rho */
  double cstd1=0.0,cstd2=0.0,cstd3=0.0; /* for calculating correlation power given nlooks */
  double defaultcorr=0.0;       /* default correlation if no correlation file */
  double rhominfactor=0.0;      /* threshold for setting unbiased correlation to 0 */

  /* pdf model parameters */
  double dzlaypeak=0.0;         /* range pdf peak for no discontinuity when bright */
  double azdzfactor=0.0;        /* fraction of dz in azimuth vs. rnage */
  double dzeifactor=0.0;        /* nonlayover dz scale factor */
  double dzeiweight=0.0;        /* weight to give dz expected from intensity */
  double dzlayfactor=0.0;       /* layover regime dz scale factor */
  double layconst=0.0;          /* normalized constant pdf of layover edge */
  double layfalloffconst=0.0;   /* factor of sigsq for layover cost increase */
  long sigsqshortmin=0;         /* min short value for costT variance */
  double sigsqlayfactor=0.0;    /* fration of ambiguityheight^2 for layover sigma */

  /* deformation mode parameters */
  double defoazdzfactor=0.0;    /* scale for azimuth ledge in defo cost function */
  double defothreshfactor=0.0;  /* factor of rho0 for discontinuity threshold */
  double defomax=0.0;           /* max discontinuity (cycles) from deformation */
  double sigsqcorr=0.0;         /* variance in measured correlation */
  double defolayconst=0.0;      /* layconst for deformation mode */

  /* algorithm parameters */
  signed char eval=0;           /* evaluate unwrapped input file if TRUE */
  signed char unwrapped=0;      /* input file is unwrapped if TRUE */
  signed char regrowconncomps=0;  /* grow connected components and exit if TRUE */
  signed char initonly=0;       /* exit after initialization if TRUE */
  signed char initmethod=0;     /* MST or MCF initialization */
  signed char costmode=0;       /* statistical cost mode */
  signed char dumpall=0;        /* dump intermediate files */
  signed char amplitude=0;      /* intensity data is amplitude, not power */
  signed char havemagnitude=0;  /* flag: create correlation from other inputs */
  signed char flipphasesign=0;  /* flag: flip phase and flow array signs */
  signed char onetilereopt=0;   /* flag: reoptimize full input after tile init */
  signed char rmtileinit=0;     /* flag to remove temporary tile unw init soln */
  long initmaxflow=0;           /* maximum flow for initialization */
  long arcmaxflowconst=0;       /* units of flow past dzmax to use for initmaxflow */
  long maxflow=0;               /* max flow for tree solve looping */
  long krowei=0, kcolei=0;      /* size of boxcar averaging window for mean ei */
  long kpardpsi=0;              /* length of boxcar for mean wrapped gradient */
  long kperpdpsi=0;             /* width of boxcar for mean wrapped gradient */
  double threshold=0.0;         /* thershold for numerical dzrcrit calculation */
  double initdzr=0.0;           /* initial dzr for numerical dzrcrit calc. (m) */
  double initdzstep=0.0;        /* initial stepsize for spatial decor slope calc. */
  double maxcost=0.0;           /* min and max float values for cost arrays */
  double costscale=0.0;         /* scale factor for discretizing to integer costs */
  double costscaleambight=0.0;  /* ambiguity height for auto costs caling */
  double dnomincangle=0.0;      /* step size for range-varying param lookup table */
  long srcrow=0,srccol=0;       /* source node location */
  double p=0.0;                 /* power for Lp-norm solution (less than 0 is MAP) */
  signed char bidirlpn=0;       /* use bidirectional Lp costs if TRUE */
  long nshortcycle=0;           /* number of points for one cycle in short int dz */
  double maxnewnodeconst=0.0;   /* number of nodes added to tree on each iteration */
  long maxnflowcycles=0;        /* max number of cycles to consider nflow done */
  double maxcyclefraction=0.0;  /* ratio of max cycles to pixels */
  long nconnnodemin=0;          /* min number of nodes to keep in connected set */
  long nmajorprune=0;           /* number of major iterations between tree pruning */
  long prunecostthresh=0;       /* cost threshold for pruning */
  long edgemasktop=0;           /* number of pixels to mask at top edge of input */
  long edgemaskbot=0;           /* number of pixels to mask at bottom edge */
  long edgemaskleft=0;          /* number of pixels to mask at left edge */
  long edgemaskright=0;         /* number of pixels to mask at right edge */
  long parentpid=0;             /* process identification number of parent */

  /* tiling parameters */
  long ntilerow=0;              /* number of tiles in azimuth */
  long ntilecol=0;              /* number of tiles in range */
  long rowovrlp=0;              /* pixels of overlap between row tiles */
  long colovrlp=0;              /* pixels of overlap between column tiles */
  long piecefirstrow=0;         /* first row (indexed from 1) for piece mode */
  long piecefirstcol=0;         /* first column (indexed from 1) for piece mode */
  long piecenrow=0;             /* number of rows for piece mode */
  long piecencol=0;             /* number of cols for piece mode */
  long tilecostthresh=0;        /* maximum cost within single reliable tile region */
  long minregionsize=0;         /* minimum number of pixels in a region */
  long nthreads=0;              /* number of parallel processes to run */
  long scndryarcflowmax=0;      /* max flow increment for which to keep cost data */
  double tileedgeweight=0.0;    /* weight applied to tile-edge secondary arc costs */
  signed char assembleonly=0;   /* flag for assemble-only (no unwrap) mode */
  signed char rmtmptile=0;      /* flag for removing temporary tile files */
  char tiledir[MAXSTRLEN]={};   /* directory for temporary tile files */

  /* connected component parameters */
  double minconncompfrac=0.0;   /* min fraction of pixels in connected component */
  long conncompthresh=0;        /* cost threshold for connected component */
  long maxncomps=0;             /* max number of connected components */
  int conncompouttype=0;        /* flag for type of connected component output file */
  
}paramT;


/* input file name data structure */
typedef struct infileST{
  char infile[MAXSTRLEN]={};          /* input interferogram */
  char magfile[MAXSTRLEN]={};         /* interferogram magnitude (optional) */
  char ampfile[MAXSTRLEN]={};         /* image amplitude or power file */
  char ampfile2[MAXSTRLEN]={};        /* second amplitude or power file */
  char weightfile[MAXSTRLEN]={};      /* arc weights */
  char corrfile[MAXSTRLEN]={};        /* correlation file */
  char estfile[MAXSTRLEN]={};         /* unwrapped estimate */
  char costinfile[MAXSTRLEN]={};      /* file from which cost data is read */
  char bytemaskfile[MAXSTRLEN]={};    /* signed char valid pixel mask */
  char dotilemaskfile[MAXSTRLEN]={};  /* signed char tile unwrap mask file */
  signed char infileformat=0;         /* input file format */
  signed char unwrappedinfileformat=0;  /* input file format if unwrapped */
  signed char magfileformat=0;        /* interferogram magnitude file format */
  signed char corrfileformat=0;       /* correlation file format */
  signed char weightfileformat=0;     /* weight file format */
  signed char ampfileformat=0;        /* amplitude file format */
  signed char estfileformat=0;        /* unwrapped-estimate file format */
}infileT;


/* output file name data structure */
typedef struct outfileST{
  char outfile[MAXSTRLEN]={};         /* unwrapped output */
  char initfile[MAXSTRLEN]={};        /* unwrapped initialization */
  char flowfile[MAXSTRLEN]={};        /* flows of unwrapped solution */
  char eifile[MAXSTRLEN]={};          /* despckled, normalized intensity */
  char rowcostfile[MAXSTRLEN]={};     /* statistical azimuth cost array */
  char colcostfile[MAXSTRLEN]={};     /* statistical range cost array */
  char mstrowcostfile[MAXSTRLEN]={};  /* scalar initialization azimuth costs */
  char mstcolcostfile[MAXSTRLEN]={};  /* scalar initialization range costs */
  char mstcostsfile[MAXSTRLEN]={};    /* scalar initialization costs (all) */
  char corrdumpfile[MAXSTRLEN]={};    /* correlation coefficient magnitude */
  char rawcorrdumpfile[MAXSTRLEN]={}; /* correlation coefficient magnitude */
  char conncompfile[MAXSTRLEN]={};    /* connected component map or mask */
  char costoutfile[MAXSTRLEN]={};     /* file to which cost data is written */
  char logfile[MAXSTRLEN]={};         /* file to which parmeters are logged */
  signed char outfileformat=0;        /* output file format */
}outfileT;


/* tile parameter data structure */
typedef struct tileparamST{
  long firstcol=0;              /* first column of tile to process (index from 0) */
  long ncol=0;                  /* number of columns in tile to process */
  long firstrow=0;              /* first row of tile to process (index from 0) */
  long nrow=0;                  /* number of rows in tile to process */
}tileparamT;


/* connectected component size structure */
typedef struct conncompsizeST{
  unsigned int tilenum=0;       /* tile index */
  unsigned int icomptile=0;     /* conn comp index in tile */
  unsigned int icompfull=0;     /* conn comp index in full array */
  long npix=0;                  /* number of pixels in conn comp */
}conncompsizeT;


/* cost-mode-specific empty structs for tag dispatch */
struct TopoCostTag { using Cost = costT; };
struct DefoCostTag { using Cost = costT; };
struct SmoothCostTag { using Cost = smoothcostT; };
struct L0CostTag { using Cost = short; };
struct L1CostTag { using Cost = short; };
struct L2CostTag { using Cost = short; };
struct LPCostTag { using Cost = short; };
struct L0BiDirCostTag { using Cost = bidircostT; };
struct L1BiDirCostTag { using Cost = bidircostT; };
struct L2BiDirCostTag { using Cost = bidircostT; };
struct LPBiDirCostTag { using Cost = bidircostT; };
struct NonGridCostTag { using Cost = Array1D<long>; };


/* type for total cost of solution (may overflow long) */
typedef double totalcostT;
#define INITTOTALCOST LARGEFLOAT


/***********************/
/* function prototypes */
/***********************/

/* functions in snaphu_tile.c */

int SetupTile(long nlines, long linelen, paramT *params,
              tileparamT *tileparams, outfileT *outfiles,
              outfileT *tileoutfiles, long tilerow, long tilecol);
Array2D<signed char> SetUpDoTileMask(infileT *infiles, long ntilerow, long ntilecol);
template<class CostTag>
int GrowRegions(Array2D<typename CostTag::Cost>& costs, Array2D<short>& flows, long nrow, long ncol,
                Array2D<incrcostT>& incrcosts, outfileT *outfiles,
                tileparamT *tileparams, paramT *params, CostTag tag);
template<class CostTag>
int GrowConnCompsMask(Array2D<typename CostTag::Cost>& costs, Array2D<short>& flows, long nrow, long ncol,
                      Array2D<incrcostT>& incrcosts, outfileT *outfiles,
                      paramT *params, CostTag tag);
template<class CostTag>
int AssembleTiles(outfileT *outfiles, paramT *params,
                  long nlines, long linelen, CostTag tag);


/* functions in snaphu_solver.c */

int SetGridNetworkFunctionPointers(void);
int SetNonGridNetworkFunctionPointers(void);
template<class CostTag>
long TreeSolve(Array2D<nodeT>& nodes, Array2D<nodesuppT>& nodesupp, nodeT *ground,
               nodeT *source, Array1D<candidateT>* candidatelistptr,
               Array1D<candidateT>* candidatebagptr, long *candidatelistsizeptr,
               long *candidatebagsizeptr, bucketT *bkts, Array2D<short>& flows,
               Array2D<typename CostTag::Cost>& costs, Array2D<incrcostT>& incrcosts, Array2D<nodeT*>& apexes,
               Array2D<signed char>& iscandidate, long ngroundarcs, long nflow,
               Array2D<float>& mag, Array2D<float>& wrappedphase, char *outfile,
               long nnoderow, Array1D<int>& nnodesperrow, long narcrow,
               Array1D<int>& narcsperrow, long nrow, long ncol,
               outfileT *outfiles, long nconnected, paramT *params, CostTag tag);
int InitNetwork(Array2D<short>& flows, long *ngroundarcsptr, long *ncycleptr,
                long *nflowdoneptr, long *mostflowptr, long *nflowptr,
                long *candidatebagsizeptr, Array1D<candidateT>* candidatebagptr,
                long *candidatelistsizeptr, Array1D<candidateT>* candidatelistptr,
                Array2D<signed char>* iscandidateptr, Array2D<nodeT*>* apexesptr,
                bucketT *bkts, long *iincrcostfileptr,
                Array2D<incrcostT>* incrcostsptr, Array2D<nodeT>* nodesptr, nodeT *ground,
                long *nnoderowptr, Array1D<int>* nnodesperrowptr, long *narcrowptr,
                Array1D<int>* narcsperrowptr, long nrow, long ncol,
                signed char *notfirstloopptr, totalcostT *totalcostptr,
                paramT *params);
long SetupTreeSolveNetwork(Array2D<nodeT>& nodes, nodeT *ground, Array2D<nodeT*>& apexes,
                           Array2D<signed char>& iscandidate, long nnoderow,
                           Array1D<int>& nnodesperrow, long narcrow, Array1D<int>& narcsperrow,
                           long nrow, long ncol);
signed char CheckMagMasking(Array2D<float>& mag, long nrow, long ncol);
int MaskNodes(long nrow, long ncol, Array2D<nodeT>& nodes, nodeT *ground,
              Array2D<float>& mag);
long MaxNonMaskFlow(Array2D<short>& flows, Array2D<float>& mag, long nrow, long ncol);
int InitNodeNums(long nrow, long ncol, Array2D<nodeT>& nodes, nodeT *ground);
int InitNodes(long nrow, long ncol, Array2D<nodeT>& nodes, nodeT *ground);
void BucketInsert(nodeT *node, long ind, bucketT *bkts);
void BucketRemove(nodeT *node, long ind, bucketT *bkts);
nodeT *ClosestNode(bucketT *bkts);
long SelectSources(Array2D<nodeT>& nodes, Array2D<float>& mag, nodeT *ground, long nflow,
                   Array2D<short>& flows, long ngroundarcs,
                   long nrow, long ncol, paramT *params,
                   Array1D<nodeT*>* sourcelistptr, Array1D<long>* nconnectedarrptr);
template<class CostTag>
long ReCalcCost(Array2D<typename CostTag::Cost>& costs, Array2D<incrcostT>& incrcosts, long flow,
                long arcrow, long arccol, long nflow, long nrow,
                paramT *params, CostTag tag);
template<class CostTag>
int SetupIncrFlowCosts(Array2D<typename CostTag::Cost>& costs, Array2D<incrcostT>& incrcosts, Array2D<short>& flows,
                       long nflow, long nrow, long narcrow,
                       Array1D<int>& narcsperrow, paramT *params, CostTag tag);
template<class CostTag>
totalcostT EvaluateTotalCost(Array2D<typename CostTag::Cost>& costs, Array2D<short>& flows, long nrow, long ncol,
                             Array1D<int>& narcsperrow, paramT *params, CostTag tag);
int MSTInitFlows(Array2D<float>& wrappedphase, Array2D<short>* flowsptr,
                 Array2D<short>& mstcosts, long nrow, long ncol,
                 Array2D<nodeT>* nodes, nodeT *ground, long maxflow);
int MCFInitFlows(Array2D<float>& wrappedphase, Array2D<short>* flowsptr, Array2D<short>& mstcosts,
                 long nrow, long ncol);


/* functions in snaphu_cost.c */
template<class CostTag>
int BuildCostArrays(Array2D<typename CostTag::Cost>* costsptr, Array2D<short>* mstcostsptr,
                    Array2D<float>& mag, Array2D<float>& wrappedphase,
                    Array2D<float>& unwrappedest, long linelen, long nlines,
                    long nrow, long ncol, paramT *params,
                    tileparamT *tileparams, infileT *infiles,
                    outfileT *outfiles, CostTag tag);
void CalcCost(Array2D<costT>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, TopoCostTag tag);
void CalcCost(Array2D<costT>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, DefoCostTag tag);
void CalcCost(Array2D<smoothcostT>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, SmoothCostTag tag);
void CalcCost(Array2D<short>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, L0CostTag tag);
void CalcCost(Array2D<short>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, L1CostTag tag);
void CalcCost(Array2D<short>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, L2CostTag tag);
void CalcCost(Array2D<short>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, LPCostTag tag);
void CalcCost(Array2D<bidircostT>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, L0BiDirCostTag tag);
void CalcCost(Array2D<bidircostT>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, L1BiDirCostTag tag);
void CalcCost(Array2D<bidircostT>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, L2BiDirCostTag tag);
void CalcCost(Array2D<bidircostT>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, LPBiDirCostTag tag);
void CalcCost(Array2D<Array1D<long>>& costs, long flow, long arcrow, long arccol,
              long nflow, long nrow, paramT *params,
              long *poscostptr, long *negcostptr, NonGridCostTag tag);
long EvalCost(Array2D<costT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, TopoCostTag tag);
long EvalCost(Array2D<costT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, DefoCostTag tag);
long EvalCost(Array2D<smoothcostT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, SmoothCostTag tag);
long EvalCost(Array2D<short>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, L0CostTag tag);
long EvalCost(Array2D<short>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, L1CostTag tag);
long EvalCost(Array2D<short>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, L2CostTag tag);
long EvalCost(Array2D<short>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, LPCostTag tag);
long EvalCost(Array2D<bidircostT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, L0BiDirCostTag tag);
long EvalCost(Array2D<bidircostT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, L1BiDirCostTag tag);
long EvalCost(Array2D<bidircostT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, L2BiDirCostTag tag);
long EvalCost(Array2D<bidircostT>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, LPBiDirCostTag tag);
long EvalCost(Array2D<Array1D<long>>& costs, Array2D<short>& flows, long arcrow, long arccol,
              long nrow, paramT *params, NonGridCostTag tag);


/* functions in snaphu_util.c */

signed char SetBooleanSignedChar(signed char *boolptr, char *str);
int WrapPhase(Array2D<float>& wrappedphase, long nrow, long ncol);
int CalcWrappedRangeDiffs(Array2D<float>& dpsi,
                          Array2D<float>& avgdpsi,
                          Array2D<float>& wrappedphase,
                          long kperpdpsi, long kpardpsi,
                          long nrow, long ncol);
int CalcWrappedAzDiffs(Array2D<float>& dpsi,
                       Array2D<float>& avgdpsi,
                       Array2D<float>& wrappedphase,
                       long kperpdpsi, long kpardpsi, long nrow, long ncol);
int CycleResidue(Array2D<float>& phase, Array2D<signed char>& residue,
                 int nrow, int ncol);
int NodeResidue(Array2D<float>& wphase, long row, long col);
int CalcFlow(Array2D<float>& phase, Array2D<short>* flowsptr, long nrow, long ncol);
int IntegratePhase(Array2D<float>& psi, Array2D<float>& phi, Array2D<short>& flows,
                   long nrow, long ncol);
Array2D<float> ExtractFlow(Array2D<float>& unwrappedphase, Array2D<short>* flowsptr,
                           long nrow, long ncol);
int FlipPhaseArraySign(Array2D<float>& arr, paramT *params, long nrow, long ncol);
int FlipFlowArraySign(Array2D<short>& arr, paramT *params, long nrow, long ncol);
int Set2DShortArray(Eigen::Ref<Array2D<short>> arr, long nrow, long ncol, long value);
signed char ValidDataArray(Array2D<float>& arr, long nrow, long ncol);
signed char NonNegDataArray(Array2D<float>& arr, long nrow, long ncol);
signed char IsFinite(double d);
long LRound(double a);
long LMin(long a, long b);
long LClip(long a, long minval, long maxval);
long Short2DRowColAbsMax(Array2D<short>& arr, long nrow, long ncol);
float LinInterp1D(Array1D<float>& arr, double index, long nelem);
float LinInterp2D(Array2D<float>& arr, double rowind, double colind,
                  long nrow, long ncol);
int Despeckle(Array2D<float>& mag, Array2D<float>* ei, long nrow, long ncol);
Array2D<float> MirrorPad(Array2D<float>& array1,
                          long nrow, long ncol, long krow, long kcol);
int BoxCarAvg(Array2D<float>& avgarr, Array2D<float>& padarr,
              long nrow, long ncol, long krow, long kcol);
char *StrNCopy(char *dest, const char *src, size_t n);
int FlattenWrappedPhase(Array2D<float>& wrappedphase, Array2D<float>& unwrappedest,
                        long nrow, long ncol);
int Add2DFloatArrays(Array2D<float>& arr1,
                     Array2D<float>& arr2,
                     long nrow, long ncol);
int StringToDouble(char *str, double *d);
int StringToLong(char *str, long *l);
int CatchSignals(void (*SigHandler)(int));
void SetDump(int signum);
void KillChildrenExit(int signum);
void SignalExit(int signum);
int StartTimers(time_t *tstart, double *cputimestart);
int DisplayElapsedTime(time_t tstart, double cputimestart);
int LongCompare(const void *c1, const void *c2);


/* functions in snaphu_io.c */

int SetDefaults(infileT *infiles, outfileT *outfiles, paramT *params);
int CheckParams(infileT *infiles, outfileT *outfiles, 
                long linelen, long nlines, paramT *params);
int ReadConfigFile(const char *conffile, infileT *infiles, outfileT *outfiles,
                   long *ncolptr, paramT *params);
int WriteConfigLogFile(infileT *infiles,
                       outfileT *outfiles, long linelen, paramT *params);
long GetNLines(infileT *infiles, long linelen, paramT *params);
int WriteOutputFile(Array2D<float>& mag,
                    Array2D<float>& unwrappedphase, char *outfile,
                    outfileT *outfiles, long nrow, long ncol);
FILE *OpenOutputFile(const char *outfile, char *realoutfile);
int Write2DArray(void **array, char *filename, long nrow, long ncol,
                 size_t size);
int Write2DRowColArray(void **array, char *filename, long nrow,
                       long ncol, size_t size);
int ReadInputFile(infileT *infiles, Array2D<float>* magptr, Array2D<float>* wrappedphaseptr,
                  Array2D<short>* flowsptr, long linelen, long nlines,
                  paramT *params, tileparamT *tileparams);
int ReadMagnitude(Array2D<float>& mag, infileT *infiles, long linelen, long nlines,
                  tileparamT *tileparams);
int ReadByteMask(Array2D<float>& mag, infileT *infiles, long linelen, long nlines,
                 tileparamT *tileparams, paramT *params);
int ReadUnwrappedEstimateFile(Array2D<float>* unwrappedestptr, infileT *infiles,
                              long linelen, long nlines,
                              paramT *params, tileparamT *tileparams);
int ReadWeightsFile(Array2D<short>* weightsptr,char *weightfile,
                    long linelen, long nlines, tileparamT *tileparams);
int ReadIntensity(Array2D<float>* pwrptr, Array2D<float>* pwr1ptr, Array2D<float>* pwr2ptr,
                  infileT *infiles, long linelen, long nlines,
                  paramT *params, tileparamT *tileparams);
int ReadCorrelation(Array2D<float>* corrptr, infileT *infiles,
                    long linelen, long nlines, tileparamT *tileparams);
int ReadAltLineFile(Array2D<float>* mag, Array2D<float>* phase, char *alfile,
                    long linelen, long nlines, tileparamT *tileparams);
int ReadAltLineFilePhase(Array2D<float>* phase, char *alfile,
                         long linelen, long nlines, tileparamT *tileparams);
int ReadComplexFile(Array2D<float>* mag, Array2D<float>* phase, char *rifile,
                    long linelen, long nlines, tileparamT *tileparams);
int ReadAltSampFile(Array2D<float>* arr1, Array2D<float>* arr2, char *infile,
                     long linelen, long nlines, tileparamT *tileparams);
int SetDumpAll(outfileT *outfiles, paramT *params);
int DumpIncrCostFiles(Array2D<incrcostT>& incrcosts, long iincrcostfile,
                      long nflow, long nrow, long ncol);
int MakeTileDir(paramT *params, outfileT *outfiles);
int ParseFilename(const char *filename, char *path, char *basename);
int SetTileInitOutfile(char *outfile, long pid);


/* function: MakeRowColArray2D()
 * -----------------------------
 * Creates a 2D array for storing arc attributes for a grid-like network. This
 * function is intended as a replacement for Get2DRowColMem() and
 * Get2DRowColZeroMem(), so the resulting array is typically treated as though
 * it were the same shape as the memory buffer created by those functions. The
 * true output array shape is 2*nrow-1 rows by ncol columns.
 */
template<typename T>
Array2D<T>
MakeRowColArray2D(long nrow, long ncol)
{
  return Array2D<T>::Zero(2*nrow-1,ncol);
}

template<>
inline Array2D<costT>
MakeRowColArray2D<costT>(long nrow, long ncol)
{
  auto arr = Array2D<costT>(2*nrow-1,ncol);

  /* fill with zeros */
  const costT val={0,0,0,0};
  std::fill_n(arr.data(),arr.size(),val);

  return arr;
}

template<>
inline Array2D<smoothcostT>
MakeRowColArray2D<smoothcostT>(long nrow, long ncol)
{
  auto arr = Array2D<smoothcostT>(2*nrow-1,ncol);

  /* fill with zeros */
  const smoothcostT val={0,0};
  std::fill_n(arr.data(),arr.size(),val);

  return arr;
}

template<>
inline Array2D<bidircostT>
MakeRowColArray2D<bidircostT>(long nrow, long ncol)
{
  auto arr = Array2D<bidircostT>(2*nrow-1,ncol);

  /* fill with zeros */
  const bidircostT val={0,0};
  std::fill_n(arr.data(),arr.size(),val);

  return arr;
}

template<>
inline Array2D<incrcostT>
MakeRowColArray2D<incrcostT>(long nrow, long ncol)
{
  auto arr = Array2D<incrcostT>(2*nrow-1,ncol);

  /* fill with zeros */
  const incrcostT val={0,0};
  std::fill_n(arr.data(),arr.size(),val);

  return arr;
}

/* function: Write2DArray() */
template<typename T>
int Write2DArray(Array2D<T>& array, const char *filename,
                 long nrow, long ncol, size_t size){

  int row;
  FILE *fp;
  char realoutfile[MAXSTRLEN]={};

  fp=OpenOutputFile(filename,realoutfile);
  for(row=0;row<nrow;row++){
    if(fwrite(array.row(row).data(),size,ncol,fp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while writing to file " + std::string(realoutfile) +
              " (device full?)");
    }
  }
  if(fclose(fp)){
    fflush(NULL);
    auto warnings=pyre::journal::warning_t("isce3.unwrap.snaphu");
    warnings << pyre::journal::at(__HERE__)
             << "WARNING: problem closing file " << realoutfile
             << " (disk full?)" << pyre::journal::endl;
  }
  return(0);
}


/* function: Write2DRowColArray() */
template<typename T>
int Write2DRowColArray(Array2D<T>& array, char *filename, long nrow,
                        long ncol, size_t size){

  int row;
  FILE *fp;
  char realoutfile[MAXSTRLEN]={};

  fp=OpenOutputFile(filename,realoutfile);
  for(row=0;row<nrow-1;row++){
    if(fwrite(array.row(row).data(),size,ncol,fp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while writing to file " + std::string(realoutfile) +
              " (device full?)");
    }
  }
  for(row=nrow-1;row<2*nrow-1;row++){
    if(fwrite(array.row(row).data(),size,ncol-1,fp)!=ncol-1){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while writing to file " + std::string(realoutfile) +
              " (device full?)");
    }
  }
  if(fclose(fp)){
    fflush(NULL);
    auto warnings=pyre::journal::warning_t("isce3.unwrap.snaphu");
    warnings << pyre::journal::at(__HERE__)
             << "WARNING: problem closing file " << realoutfile << " (disk full?)"
             << pyre::journal::endl;
  }
  return(0);
}


/* function: Read2DArray() */
template<typename T>
int Read2DArray(Array2D<T>* arr, char *infile, long linelen, long nlines,
                tileparamT *tileparams, size_t /*elptrsize*/, size_t elsize){

  FILE *fp;
  long filesize,row,nrow,ncol,padlen;

  /* open the file */
  if((fp=fopen(infile,"r"))==NULL){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Can't open file " + std::string(infile));
  }

  /* get number of lines based on file size and line length */ 
  fseek(fp,0,SEEK_END);
  filesize=ftell(fp);
  if(filesize!=(nlines*linelen*elsize)){
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
  if(!arr->size()){
    *arr=Array2D<T>(nrow,ncol);
  }

  /* read the data */
  fseek(fp,(linelen*tileparams->firstrow+tileparams->firstcol)
        *elsize,SEEK_CUR);
  padlen=(linelen-ncol)*elsize;
  for(row=0;row<nrow;row++){
    if(fread(arr->row(row).data(),elsize,ncol,fp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while reading from file " + std::string(infile));
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);

  /* done */
  return(0);

}


/* function: Read2DRowColFile() */
template<typename T>
int Read2DRowColFile(Array2D<T>* arr, char *filename, long linelen, long nlines,
                     tileparamT *tileparams, size_t size){

  FILE *fp;
  long row, nel, nrow, ncol, padlen, filelen;
 
  /* open the file */
  if((fp=fopen(filename,"r"))==NULL){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Can't open file " + std::string(filename));
  }

  /* get number of data elements in file */ 
  fseek(fp,0,SEEK_END);
  filelen=ftell(fp);
  fseek(fp,0,SEEK_SET);
  nel=(long )(filelen/size);

  /* check file size */
  if(2*linelen*nlines-nlines-linelen != nel || (filelen % size)){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "File " + std::string(filename) + " wrong size (" +
            std::to_string(2*linelen*nlines-nlines-linelen) +
            " elements expected)");
  }

  /* get memory if passed pointer is NULL */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(!arr->size()){
    *arr=MakeRowColArray2D<T>(nrow,ncol);
  }

  /* read arrays */
  fseek(fp,(linelen*tileparams->firstrow+tileparams->firstcol)
        *size,SEEK_SET);
  padlen=(linelen-ncol)*size;
  for(row=0; row<nrow-1; row++){
    if(fread(arr->row(row).data(),size,ncol,fp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while reading from file " + std::string(filename));
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fseek(fp,(linelen*(nlines-1)+(linelen-1)*tileparams->firstrow
            +tileparams->firstcol)*size,SEEK_SET);
  for(row=nrow-1; row<2*nrow-1; row++){
    if(fread(arr->row(row).data(),size,ncol-1,fp)!=ncol-1){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while reading from file " + std::string(filename));
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);

  /* done */
  return(0);

}


/* function: Read2DRowColFileRows()
 * --------------------------------
 * Similar to Read2DRowColFile(), except reads only row (horizontal) data
 * at specified locations.  tileparams->nrow is treated as the number of
 * rows of data to be read from the RowCol file, not the number of 
 * equivalent rows in the orginal pixel file (whose arcs are represented
 * in the RowCol file).
 */
template<typename T>
int Read2DRowColFileRows(Array2D<T>* arr, char *filename, long linelen, 
                         long nlines, tileparamT *tileparams, size_t size){

  FILE *fp;
  long row, nel, nrow, ncol, padlen, filelen;
 
  /* open the file */
  if((fp=fopen(filename,"r"))==NULL){
    fflush(NULL);
    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "Can't open file " + std::string(filename));
  }

  /* get number of data elements in file */ 
  fseek(fp,0,SEEK_END);
  filelen=ftell(fp);
  fseek(fp,0,SEEK_SET);
  nel=(long )(filelen/size);

  /* check file size */
  if(2*linelen*nlines-nlines-linelen != nel || (filelen % size)){
    fflush(NULL);
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            "File " + std::string(filename) + " wrong size (" +
            std::to_string(2*linelen*nlines-nlines-linelen) +
            " elements expected)");
  }

  /* get memory if passed pointer is NULL */
  nrow=tileparams->nrow;
  ncol=tileparams->ncol;
  if(!arr->size()){
    *arr=MakeRowColArray2D<T>(nrow,ncol);
  }

  /* read arrays */
  fseek(fp,(linelen*tileparams->firstrow+tileparams->firstcol)
        *size,SEEK_SET);
  padlen=(linelen-ncol)*size;
  for(row=0; row<nrow; row++){
    if(fread(arr->row(row).data(),size,ncol,fp)!=ncol){
      fflush(NULL);
      throw isce3::except::RuntimeError(ISCE_SRCINFO(),
              "Error while reading from file " + std::string(filename));
    }
    fseek(fp,padlen,SEEK_CUR);
  }
  fclose(fp);

  /* done */
  return(0);

}


/*******************************************/
/* global (external) variable declarations */
/*******************************************/

/* flags used for signal handling */
extern char dumpresults_global;
extern char requestedstop_global;

/* node pointer for marking arc not on tree in apex array */
/* this should be treat as a constant */
extern nodeT NONTREEARC[1];

} // namespace isce3::unwrap
