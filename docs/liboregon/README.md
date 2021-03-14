A Matlab toolkit for Recursive Bayesian Estimation. This software consolidates research on new methods for recursive Bayesian estimation and Kalman filtering by Rudolph van der Merwe and Eric A. Wan. The code is developed and maintained by Rudolph van der Merwe at the OGI School of Science & Engineering at OHSU (Oregon Health & Science University). This document was last modified on : September 5th, 2006. UPDATE MATLABPATH : Add 'core' and 'netlab' directories. netlab is needed to use any of the GMM based particle filter and hybrid algorithms. see Chapters 5 and 7 of "Kalman Filtering and Neural Networks" edited by Simon Haykin and published by Wiley. These cover dual extended Kalman filter methods and the Unscented Kalman Filter in great detail.

- Kalman filter
- Extended Kalman filter
- Sigma-Point Kalman filters (SPKF), Unscented Kalman filter (UKF), Central difference Kalman filter (CDKF), Square-root SPKFs (SRUKF, SRCDKF), Gaussian mixture SPKFs  (not implemented yet)
- Particle filters, Generic particle filter, Sigma-Point particle filter, Gaussian sum particle filter, Gaussian mixture sigma-point particle filter (beta), Auxiliary sigma-point particle filter (alpha)

designed to allow reuse of a state space definition for state, parameter and joint estimation, using a variety of different inference algorithms. you define your system once in a higher-level state space framework, and then the inference framework generator *geninfds* together with the inference system noise source generator *gensysnoiseds* will adapt/remap that model into a lower-level state space framework needed for whatever type of estimation you want to do. This allows you to focus on defining the model, doing data IO, etc. without having to get bogged down into casting the problem into a different framework each time you want to use a different estimator or want to change the type of inference your doing. I.e. the internal inference implementation is hidden or as transparent as possible with respect to the problem definition by the user.

has certain 'OOP' like behavior, this was implemented using nothing more complex than Matlab's built-in structure data types as well as function handles. This has certain advantages as well as disadvantages. The OOP-like features of the inference system is needed to implement the abstraction layer which again allow the 'define-once use-many-times-in-different-ways' nature as described in Section 1 above. The downside is that one pays a speed penalty using this abstraction approach in Matlab, due to the cost of function calls and the lack of true pass-by-reference inherent in Matlab. The upside is that the toolkit should be general enough and applicable to a large group of different probabilistic inference problems which can be cast into a recursive state space framework.

Here follow a bare bones, top level 'check list' of steps to follow to use  for your own estimation problems. If it doesn't make sense, consult the well documented example code as well as the built-in help information for all the  functions, i.e. 'help srukf', etc. Copy the general state space model *gssm* template and adapt it to your own specific problem. This file is called 'gssm.m' and it resides in the core code directory. define the needed dimension variables (i.e. state, observations, process and measurement noise, exogenous inputs, etc.) define and embed a noise source data structure for both the process noise source and the observation noise source. Use the 'gennoiseds' function for this purpose. define a number of required (and some optional, depending on the type of inference algorithm that will be used) subfunctions: These include the state transition function, the state observation function, and a function that is used to set the internal parameters of the model. Make sure that your problem specific GSSM file preserves the required interface for not only the main gateway function, but also for the required subfunctions. The base correctness of a user defined GSSM file can be tested using the 'consistent' function. 

Initialize an instance of your model by calling the customized GSSM file with the 'init' argument, i.e. If you copied gssm.m and adapted it to your problem, say calling the new file 'my_gssm.m', issue the following call: model = my_special_gssm('init'); Generate an Inference Data Structure *InferenceDS* from this instantiated model data structure as well as further estimation algorithm specific information. This is done through a call to the *geninfds* function, i.e. InfDS = geninfds(Arg) where Arg is and argument data structure that contains (as fields) all the needed arguments to build the InferenceDS object. I.e. Arg.model = model; % GSSM model, Arg.type  = 'state'; % inference type. Arg.type is a string that specifies what type of inference will be done on the system, i.e. state estimation = 'state', parameter estimation = 'parameter', joint estimation = 'joint'. The required fields of Arg is specified in the built-in help info for 'geninfds', i.e. 'help geninfds'. 

Generate the needed process and observation noise sources needed by the inference algorithms, by passing InfDS to the 'gensysnoiseds' function. See 'help gensysnoiseds' for the arguments. This will return two noise source data structure (NoiseDS). One for the system process noise and one for the system observation noise. You also need to pass to this function the name of the inference algorithm which will be used, since the internal form/structure of the noise sources depend on this, i.e. [pNoise, oNoise, InfDS] = gensysnoiseds(InfDS, 'srcdkf'); would generate the needed process and observation noise sources needed to use a Square-Root Central Difference Kalman Filter (SRCDKF) for inference on the system defined by InfDS. Add any inference algorithm specific parameters as auxiliary fields to the InfDS inference data structure object. E.g., in order to use any of the SPKF algorithms (ukf, cdkf, srcdkf, srukf), you need to add the '.spkfParams' field to InfDS and use it to pass the needed SPKF scaling parameters to the function, i.e. for the CDKF InfDS.spkfParams = sqrt(3);   % optimal for Gaussian priors

Pass the Inference Data Structure *InfDS* from step 3 as well as the system noise sources (pNoise & oNoise) from step 4 to the specific inference algorithm together with the relevant data and further algorithm specific variables. (See the built-in help for each inference algorithm for more detail). Note : All the inference algorithms can operate in batch mode, i.e. you pass it a sequence of observations along with the initial conditions, and the filter will then internally iterate through the sequence, returning the corresponding estimated sequence, OR they can operate on a frame by frame (online) basis. See examples.

Inference for models that use angular quantities for either state or observation vector components: If your system's state space model has state or observation vector components that are angular quantities that are measured in radians, it can lead to discontinuity problems at the +- Pi radians boundary. This is especially important when using a SPKF based algorithm (ukf, cdkf, srukf, srcdkf & sppf) for inference. To facilitate correct inference for such cases, you need to define an extra field in your GSSM data structure that contains an index vector (vector of vector component indices) indicating which of the state vector components are angular quantities and which of the observation vector components are angular quantities. These field are called '.stateAngleCompIdxVec' for the state vector and '.obsAngleCompIdxVec' for the observation vector. E.g. say the 2nd and 4th components of your state vector and the 1st component of your observation vector are angular quantities. You would then set the following fields inside the 'initialize' section of your custom GSSM file gssm.stateAngleCompIdxVec = [2 4]; gssm.obsAngleCompIdxVec   = [1]; These fields can be omitted completely if there are no angular components in either the state or observation vectors. See examples/state_estimation/demse4 for a demonstration of how this is used for a bearings-only tracking problem.

Process noise adaptation for parameter- and joint estimation. In order to speed up convergence speed and improve final estimation error, it is recommended to adapt the variance of the process noise source during parameter estimation (the same goes for joint estimation, but only relating to the subvector of the process noise affecting the parameter components of the state vector). This process noise adaptation technique is supported by all the  inference algorithms and can be accessed by setting the following fields in the process noise data structure. pNoise.adaptMethod    and   pNoise.adaptParams The first field is a string indicating the type of adaptation and the second field is a vector of parameters needed by the specific adaptation method. The options (together with their respective parameters) are: .adaptMethod = 'anneal' : Annealing, .adaptParams = [annealing_factor minimum_allowed_variance], .adaptMethod = 'lambda-decay' :  RLS like exponential decay, .adaptParams = [lambda_factor minimum_allowed_variance], .adaptMethod = 'robbins-monro' :  Robbins-Monro Stochastic approximation, .adaptParams = [1/nu_initial 1/nu_final]. When adapting the process noise source, it is important to return the process noise data structure to the calling routine (see parameter and joint estimation examples). For more information on how these different methods work, consult the publications section on the  website (there is a thorough discussion in the unofficial accompanying text to , "Kalman Filters and Neural Networks" by Simon Haykin.
     