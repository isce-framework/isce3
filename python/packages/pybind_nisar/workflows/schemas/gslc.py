# goofy but portable

schema = """
runconfig:
    name: str()

    groups:
        PGENameGroup:
            PGEName: enum('GSLC_L_PGE')
       
        InputFileGroup:
            # REQUIRED - One NISAR L1B RSLC formatted HDF5 file
            InputFilePath: list(str(), min=1, max=1)
       
        DynamicAncillaryFileGroup:
            # Digital elevation model
            DEMFile: str()
       
        ProductPathGroup:
            # Directory where PGE will place results
            ProductPath: str()
       
            # Directory where SAS can write temporary data
            ScratchPath: str()
       
            # Intermediate file name.  SAS writes output product to the following file.
            # After the SAS completes, the PGE wrapper renames the product file
            # according to proper file naming conventions.
            SASOutputFile: str()

        PrimaryExecutable:
            ProductType: enum('GSLC')

        DebugLevelGroup:
            DebugSwitch: bool()
       
        Geometry:
            CycleNumber: int(min=1, max=999)
            RelativeOrbitNumber: int(min=1, max=173)
            FrameNumber: int(min=1, max=176)
            OrbitDirection: enum('Descending', 'Ascending')
       
        #adt section - isce3 + pyre workflow
        processing:
            # Mechanism to select frequencies and polarizations
            input_subset:
                # List of frequencies to process. Default empty representing all
                list_of_frequencies:
                    # valid values for polarizations
                    # empty for all polarizations found in RSLC
                    # [polarizations] for list of specific frequency(s) e.g. [HH, HV] or [HH]
                    A: any(list(str(min=2, max=2), min=1, max=4), str(min=2, max=2), null(), required=False)
                    B: any(list(str(min=2, max=2), min=1, max=4), str(min=2, max=2), null(), required=False)
        
            # Only checked when internet access is available
            dem_download:
                # s3 bucket / curl URL / local file
                source: str(required=False)
                top_left:
                    x: num(required=False)
                    y: num(required=False)
               
                bottom_right:
                    x: num(required=False)
                    y: num(required=False)
        
            # Pre-processing options (before geocoding)
            pre_process:
                filter:
                    A:
                        type: str(required=False)
                        parameters: str(required=False)
                    B:
                        type: str(required=False)
                        parameters: str(required=False)

            # Mechanism to specify output posting and DEM
            geocode:
                # Same as input DEM if not provided.
                outputEPSG: int(required=False)
               
                # Output posting in same units as output EPSG. Single value or list indicating the output posting for each frequency
                output_posting:
                    A:
                        x_posting: num(min=0, required=False)
                        y_posting: num(min=0, required=False)
                    B:
                        x_posting: num(min=0, required=False)
                        y_posting: num(min=0, required=False)
               
                # To control output grid in same units as output EPSG
                x_snap: num(min=0, required=False)
               
                # To control output grid in same units as output EPSG
                y_snap: num(min=0, required=False)
               
                top_left:    
                    # Set top-left y in same units as output EPSG
                    y_abs: num(required=False)
                   
                    # Set top-left x in same units as output EPSG
                    x_abs: num(required=False)
               
                bottom_right:  
                    # Set bottom-right y in same units as output EPSG
                    y_abs: num(required=False)
                   
                    # Set bottom-right x in same units as output EPSG
                    x_abs: num(required=False)

            geo2rdr:
                threshold: num(min=1.0e-9, max=1.0e-3, required=False)
                maxiter: int(min=10, max=50, required=False)

            blocksize:
                y: int(min=100, max=10000)

            dem_margin: num()

            flatten: bool()
        
        # To setup type of worker
        worker:
            # To prevent downloading DEM / other data automatically. Default True
            internet_access: bool(required=False)
           
            # To explicitly use GPU capability if available. Default False
            gpu_enabled: bool(required=False)

"""
