        --  ReBEL : Recursive Bayesian Estimation Library --

         A Matlab toolkit for Recursive Bayesian Estimation

        Copyright 2006, Oregon Health and Science University


1) WHAT IS ReBEL ?

ReBEL is a Matlab� toolkit of functions and scripts, designed to
facilitate sequential Bayesian inference (estimation) in general state
space models. This software consolidates research on new methods for
recursive Bayesian estimation and Kalman filtering by Rudolph van der
Merwe and Eric A. Wan. The code is developed and maintained by Rudolph
van der Merwe at the OGI School of Science & Engineering at OHSU
(Oregon Health & Science University). 

ReBEL is a 'work in progress' and currently contains most of the
following functional units which can be used for state-, parameter-
and joint-estimation: 

    * Kalman filter
    * Extended Kalman filter
    * Sigma-Point Kalman filters (SPKF)
          o Unscented Kalman filter (UKF)
          o Central difference Kalman filter (CDKF)
          o Square-root SPKFs (SRUKF, SRCDKF)
          o Gaussian mixture SPKFs  (not implemented yet)
    * Particle filters
          o Generic particle filter
          o Sigma-Point particle filter
          o Gaussian sum particle filter
          o Gaussian mixture sigma-point particle filter (beta)
          o Auxiliary sigma-point particle filter (alpha)

The code is designed to be as general, modular and extensible as
possible, while at the same time trying to be as computationally
efficient as possible. It has been tested with Matlab 7.2 (R2006a). 


2) INSTALLATION

Unpack the ReBEL distribution archive using either 'tar -xzvf
ReBEL-X.Y.Z.tar.gz' under Unix or unzip the ReBEL-X.Y.Z.zip archive under
Windows. This will create the 'ReBEL-X.Y.Z' directory tree in the current
directory. 


3) UPDATE MATLABPATH : Add 'core' and 'netlab' directories

Make sure the 'ReBEL-X.Y.Z/core/' subdirectory is in the Matlab
path by either setting the MATLABPATH environment variable accordingly
or using the 'path' command from the Matlab command line.

(X = major release number , Y = minor release number , Z = buxfix version
number)

In order to use the bundled Netlab neural network toolkit, also add
the 'ReBEL-X.Y.Z/netlab/' subdirectory to your MATLABPATH. This is needed
to use any of the GMM based particle filter and hybrid algorithms.


4) DOCUMENTATION

Since ReBEL is an constantly 'evolving' research tool used in the
Machine Learning and Signal Processing Group ( http://choosh.csee.ogi.edu/index.html
), the documentation usually lags behind a bit. For the most up to date
documentation, refer to the online documentation section of the ReBEL
project website (see below). A small 'Quick start Guide' is also
provided. This together with the (hopefully) well documented code 
examples in the 'examples' subdirectory will serve as the initial
minimal set of documentation for the alpha release of ReBEL.

The unofficial accompanying "text" to ReBEL are Chapters 5 and 7 of
"Kalman Filtering and Neural Networks" edited by Simon Haykin and
published by Wiley. These cover dual extended Kalman filter methods
and the Unscented Kalman Filter in great detail. Other algorithms
included in ReBEL are also covered in the rest of the book. See the
ReBEL website for more detail.


5) WEBSITE

For latest information and online documentation, always visit
the project website at http://choosh.csee.ogi.edu/rebel/ 


6) MAILING LIST AND USER DISCUSSION BOARD

There is a discussion group for users and developers of ReBEL hosted
at Yahoo Groups. Here is the URL:

http://groups.yahoo.com/group/rebel_toolkit/


-----------------------------------------------------------------------
This document was last modified on : September 5th, 2006.

Quick-Start Guide for ReBEL Toolkit
===================================

The purpose of this guide, together with the hopefully well commented
code functions and code examples (in 'examples' subdirectory) is to
act as a minimal set of documentation to 'get you going' using the
ReBEL toolkit. The official documentation for the toolkit will
hopefully evolve over time into a more complete reference guide.


1. ReBEL Design Philosophy

ReBEL was designed to allow reuse of a system state space definition
for state, parameter and joint estimation, using a variety of
different inference algorithms. In other words, you define your system
once in a standard general state space framework, and then the ReBEL inference
framework generator (geninfds) together with the inference system
noise source generator (gensysnoiseds) will adapt/remap that model
into the relevant state space framework needed for whatever type of
estimation you want to do.

This allows you to focus on defining the model, doing data IO,
etc. without having to get bogged down into casting the problem into a
different framework each time you want to use a different estimator or
want to change the type of inference your doing. I.e. the internal
inference implementation is hidden or as transparent as possible with
respect to the problem definition by the user.


2. ReBEL Implementation Philosophy

Even though ReBEL has certain 'OOP' like behavior, this was
implemented using nothing more complex than Matlab's built-in
structure data types as well as function handles. This has certain
advantages as well as disadvantages. The OOP-like features of the ReBEL
inference system is needed to implement the abstraction layer which
again allow the 'define-once use-many-times-in-different-ways' nature
as described in Section 1 above. The downside is that one pays a speed
penalty using this abstraction approach in Matlab, due to the cost of
function calls and the lack of true pass-by-reference inherent in
Matlab. The upside is that the toolkit should be general enough and
applicable to a large group of different probabilistic inference
problems which can be cast into a recursive state space framework.


3. Steps in using ReBEL for an estimation problem   (See example code)

Here follow a bare bones, top level 'check list' of steps to follow to
use ReBEL for your own estimation problems. If it doesn't make sense,
consult the well documented example code as well as the built-in help
information for all the ReBEL functions, i.e. 'help srukf', etc.

	1) Copy the general state space model (gssm) template and adapt it
       to your own specific problem. This file is called 'gssm.m' and it
       resides in the core code directory.

         - You have to define the needed dimension variables
           (i.e. state, observations, process and measurement noise, 
            exogenous inputs, etc.)
 
         - You have to define and embed a noise source data structure
           for both the process noise source and the observation noise
           source. Use the 'gennoiseds' function for this purpose.  

         - You have to define a number of required (and some optional,
           depending on the type of inference algorithm that will be
           used) subfunctions: These include the state transition
           function, the state observation function, and a function
           that is used to set the internal parameters of the model.

         - Make sure that your problem specific GSSM file preserves
           the required interface for not only the main gateway
           function, but also for the required subfunctions.

         - The base correctness of a user defined GSSM file can be tested
           using the 'consistent' function.


     2) Initialize an instance of your model by calling the customized
        GSSM file with the 'init' argument, i.e. If you copied gssm.m
        and adapted it to your problem, say calling the new file
        'my_gssm.m', issue the following call:

                 model = my_special_gssm('init');


     3) Generate an Inference Data Structure (InferenceDS) from this
        instantiated model data structure as well as further
        estimation algorithm specific information. This is done
        through a call to the 'geninfds' function, i.e.

                 InfDS = geninfds(Arg)

        where Arg is and argument data structure that contains (as
        fields) all the needed arguments to build the InferenceDS
        object. I.e.
    
                Arg.model = model;       % GSSM model
                Arg.type  = 'state';     % inference type
                        
        Arg.type is a string that specifies what type of inference
        will be done on the system, i.e.

                state estimation = 'state'
                parameter estimation = 'parameter'
                joint estimation = 'joint'
                 
        The required fields of Arg is specified in the built-in help
        info for 'geninfds', i.e. 'help geninfds'. 
        
        See the example code for more detail.


     4) Generate the needed process and observation noise sources
        needed by the inference algorithms, by passing InfDS to the
        'gensysnoiseds' function. See 'help gensysnoiseds' for the
        arguments. This will return two noise source data structure
        (NoiseDS). One for the system process noise and one for the
        system observation noise. You also need to pass to this
        function the name of the inference algorithm which will be
        used, since the internal form/structure of the noise sources 
        depend on this, i.e.

          [pNoise, oNoise, InfDS] = gensysnoiseds(InfDS, 'srcdkf');

        would generate the needed process and observation noise
        sources needed to use a Square-Root Central Difference Kalman
        Filter (SRCDKF) for inference on the system defined by InfDS.

     5) Add any inference algorithm specific parameters as auxiliary
        fields to the InfDS inference data structure object. E.g., in
        order to use any of the SPKF algorithms (ukf, cdkf, srcdkf,
        srukf), you need to add the '.spkfParams' field to InfDS and
        use it to pass the needed SPKF scaling parameters to the
        function, i.e. for the CDKF

            InfDS.spkfParams = sqrt(3);   % optimal for Gaussian priors


     6) Pass the Inference Data Structure (InfDS) from step 3 as well
        as the system noise sources (pNoise & oNoise) from step 4 to
        the specific inference algorithm together with the relevant
        data and further algorithm specific variables. (See the
        built-in help for each inference algorithm for more detail).
 
        Note : All the inference algorithms can operate in batch mode,
        i.e. you pass it a sequence of observations along with the
        initial conditions, and the filter will then internally
        iterate through the sequence, returning the corresponding
        estimated sequence, OR they can operate on a frame by frame
        (online) basis. See examples.

     

4) OTHER ISSUES AND UNDOCUMENTED FEATURES

This section is a 'catch-all' for some things that you need to know
about, but which haven't been documented elsewhere yet.

  1) Inference for models that use angular quantities for either
     state or observation vector components:

     If your system's state space model has state or observation
     vector components that are angular quantities that are measured
     in radians, it can lead to discontinuity problems at the +- Pi
     radians boundary. This is especially important when using a SPKF
     based algorithm (ukf, cdkf, srukf, srcdkf & sppf) for inference.
     To facilitate correct inference for such cases, you need to
     define an extra field in your GSSM data structure that contains an
     index vector (vector of vector component indices) indicating
     which of the state vector components are angular quantities and which
     of the observation vector components are angular quantities.
     
     These field are called '.stateAngleCompIdxVec' for the state
     vector and '.obsAngleCompIdxVec' for the observation vector.

     E.g. say the 2nd and 4th components of your state vector and the
          1st component of your observation vector are angular
          quantities. You would then set the following fields inside
          the 'initialize' section of your custom GSSM file

          gssm.stateAngleCompIdxVec = [2 4];
          gssm.obsAngleCompIdxVec   = [1];

     These fields can be omitted completely if there are no angular
     components in either the state or observation vectors.

     See examples/state_estimation/demse4 for a demonstration of how
     this is used for a bearings-only tracking problem.

  
  2) Process noise adaptation for parameter- and joint estimation.

     In order to speed up convergence speed and improve final
     estimation error, it is recommended to adapt the variance of the
     process noise source during parameter estimation (the same goes
     for joint estimation, but only relating to the subvector of the 
     process noise affecting the parameter components of the state
     vector). 

     This process noise adaptation technique is supported by all the
     ReBEL inference algorithms and can be accessed by setting the
     following fields in the process noise data structure.

        pNoise.adaptMethod    and   pNoise.adaptParams

     The first field is a string indicating the type of adaptation and
     the second field is a vector of parameters needed by the specific
     adaptation method. The options (together with their respective
     parameters) are: 

       a)  .adaptMethod = 'anneal'   :   Annealing
           .adaptParams = [annealing_factor minimum_allowed_variance]

       b)  .adaptMethod = 'lambda-decay' :  RLS like exponential decay
           .adaptParams = [lambda_factor minimum_allowed_variance]

       c)  .adaptMethod = 'robbins-monro' :  Robbins-Monro Stochastic 
                                             approximation
           .adaptParams = [1/nu_initial 1/nu_final]

     When adapting the process noise source, it is important to return
     the process noise data structure to the calling routine (see
     parameter and joint estimation examples).

     For more information on how these different methods work, consult
     the publications section on the ReBEL website (there is a
     thorough discussion in the unofficial accompanying text to ReBEL,
     "Kalman Filters and Neural Networks" by Simon Haykin.
     
    
 


========================================================================     
     



