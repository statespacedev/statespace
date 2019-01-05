function [xh, Sx, pNoise, oNoise, InternalVariablesDS] = srukf(state, Sstate, pNoise, oNoise, obs, U1, U2, InferenceDS)

% SRUKF  Square Root Unscented Kalman Filter (Sigma-Point Kalman Filter variant)
%
%   [xh, Sx, pNoise, oNoise, InternalVariablesDS] = SRUKF(state, Sstate, pNoise, oNoise, obs, U1, U2, InferenceDS)
%
%   This filter assumes the following standard state-space model:
%
%     x(k) = ffun[x(k-1),v(k-1),U1(k-1)]
%     y(k) = hfun[x(k),n(k),U2(k)]
%
%   where x is the system state, v the process noise, n the observation noise, u1 the exogenous input to the state
%   transition function, u2 the exogenous input to the state observation function and y the noisy observation of the
%   system.
%
%   INPUT
%         state                state mean at time k-1          ( xh(k-1) )
%         Sstate               lower triangular Cholesky factor of state covariance at time k-1    ( Sx(k-1) )
%         pNoise               process noise data structure     (must be of type 'gaussian' or 'combo-gaussian')
%         oNoise               observation noise data structure (must be of type 'gaussian' or 'combo-gaussian')
%         obs                  noisy observations starting at time k ( y(k),y(k+1),...,y(k+N-1) )
%         U1                   exogenous input to state transition function starting at time k-1 ( u1(k-1),u1(k),...,u1(k+N-2) )
%         U2                   exogenous input to state observation function starting at time k  ( u2(k),u2(k+1),...,u2(k+N-1) )
%         InferenceDS          SPOK inference data structure generated by GENINFDS function.
%
%   OUTPUT
%         xh                   estimates of state starting at time k ( E[x(t)|y(1),y(2),...,y(t)] for t=k,k+1,...,k+N-1 )
%         Sx                   Cholesky factor of state covariance at time k  ( Sx(k) )
%         pNoise               process noise data structure     (possibly updated)
%         oNoise               observation noise data structure (possibly updated)
%
%         InternalVariablesDS  (optional) internal variables data structure
%            .xh_                 predicted state mean ( E[x(t)|y(1),y(2),..y(t-1)] for t=k,k+1,...,k+N-1 )
%            .Sx_                 predicted state covariance (Cholesky factor)
%            .yh_                 predicted observation ( E[y(k)|Y(k-1)] )
%            .inov                inovation signal
%            .Pinov               inovation covariance
%            .KG                  Kalman gain
%
%   Required InferenceDS fields:
%         .spkfParams           SPKF parameters = [alpha beta kappa] with
%                                   alpha  :  UKF scale factor
%                                   beta   :  UKF covariance correction factor
%                                   kappa  :  UKF secondary scaling parameter
%   Copyright (c) Oregon Health & Science University (2006)
%
%   This file is part of the ReBEL Toolkit. The ReBEL Toolkit is available free for
%   academic use only (see included license file) and can be obtained from
%   http://choosh.csee.ogi.edu/rebel/.  Businesses wishing to obtain a copy of the
%   software should contact rebel@csee.ogi.edu for commercial licensing information.
%
%   See LICENSE (which should be part of the main toolkit distribution) for more
%   detail.

%=============================================================================================

Xdim  = InferenceDS.statedim;                                % extract state dimension
Odim  = InferenceDS.obsdim;                                  % extract observation dimension
U1dim = InferenceDS.U1dim;                                   % extract exogenous input 1 dimension
U2dim = InferenceDS.U2dim;                                   % extract exogenous input 2 dimension
Vdim  = InferenceDS.Vdim;                                    % extract process noise dimension
Ndim  = InferenceDS.Ndim;                                    % extract observation noise dimension

NOV = size(obs,2);                                           % number of input vectors

%------------------------------------------------------------------------------------------------------------------
%-- ERROR CHECKING

if (nargin ~= 8) error(' [ srukf ] Not enough input arguments.'); end

if (Xdim~=size(state,1)) error('[ srukf ] Prior state dimension differs from InferenceDS.statedim'); end
if (Xdim~=size(Sstate,1)) error('[ srukf ] Prior state covariance dimension differs from InferenceDS.statedim'); end
if (Odim~=size(obs,1)) error('[ srukf ] Observation dimension differs from InferenceDS.obsdim'); end
if U1dim
  [dim,nop] = size(U1);
  if (U1dim~=dim) error('[ srukf ] Exogenous input U1 dimension differs from InferenceDS.U1dim'); end
  if (dim & (NOV~=nop)) error('[ srukf ] Number of observation vectors and number of exogenous input U1 vectors do not agree!'); end
end
if U2dim
  [dim,nop] = size(U2);
  if (U2dim~=dim) error('[ srukf ] Exogenous input U2 dimension differs from InferenceDS.U2dim'); end
  if (dim & (NOV~=nop)) error('[ srukf ] Number of observation vectors and number of exogenous input U2 vectors do not agree!'); end
end

%--------------------------------------------------------------------------------------------------------------

% setup buffer
xh   = zeros(Xdim,NOV);
xh_  = zeros(Xdim,NOV);
yh_  = zeros(Odim,NOV);
inov = zeros(Odim,NOV);

% Get UKF scaling parameters
alpha = InferenceDS.spkfParams(1);
beta  = InferenceDS.spkfParams(2);
kappa = InferenceDS.spkfParams(3);

% Get index vectors for any of the state or observation vector components that are angular quantities
% which have discontinuities at +- Pi radians ?

sA_IdxVec = InferenceDS.stateAngleCompIdxVec;
oA_IdxVec = InferenceDS.obsAngleCompIdxVec;


switch  InferenceDS.inftype

%======================================= PARAMETER ESTIMATION VERSION ===========================================
case 'parameter'

    L = Xdim+Ndim;                                  % augmented state dimension
    nsp = 2*L+1;                                    % number of sigma-points
    kappa = alpha^2*(L+kappa)-L;                    % compound scaling parameter

    W = [kappa 0.5 0]/(L+kappa);                    % sigma-point weights
    W(3) = W(1) + (1-alpha^2) + beta;

    sqrtW = W;
    possitive_W3 = (W(3) > 0);                      % is zero'th covariance weight possitive?
    sqrtW(1:2) = sqrt(W(1:2));                      % square root weights
    sqrtW(3) = sqrt(abs(W(3)));

    Sqrt_L_plus_kappa = sqrt(L+kappa);

    Zeros_Xdim_X_Ndim     = zeros(Xdim,Ndim);
    Zeros_Ndim_X_Xdim     = zeros(Ndim,Xdim);

    Sv = pNoise.cov;
    dv = diag(Sv);
    Sn = oNoise.cov;
    mu_n = oNoise.mu;

    Sx = Sstate;

    %---  Loop over all input vectors ---
    for i=1:NOV,

        UU2 = cvecrep(U2(:,i),nsp);

        %------------------------------------------------------
        % TIME UPDATE

        xh_(:,i) = state;

        if pNoise.adaptMethod
        %--------------------------------------------
            switch pNoise.adaptMethod

            case 'lambda-decay'
                Sx_ = sqrt(pNoise.adaptParams(1))*Sx;

            case {'anneal','robbins-monro'}
                Sx_ = Sx + Sv;

            end
        %---------------------------------------------
        else
            Sx_ = Sx;
        end

        Z   = cvecrep([xh_(:,i); mu_n],nsp);
        Sz  = [Sx_ Zeros_Xdim_X_Ndim; Zeros_Ndim_X_Xdim Sn];
        sS  = Sqrt_L_plus_kappa * Sz;
        Z(:,2:nsp) = Z(:,2:nsp) + [sS -sS];

        Y_ = InferenceDS.hfun( InferenceDS, Z(1:Xdim,:), Z(Xdim+1:Xdim+Ndim,:), UU2);

        temp1 = Z(1:Xdim,:) - cvecrep(xh_(:,i),nsp);


        %-- Calculate predicted observation mean, dealing with angular discontinuities if needed
        if isempty(oA_IdxVec)
            yh_(:,i) = W(1)*Y_(:,1) + W(2)*sum(Y_(:,2:nsp),2);
            temp2 = Y_ - cvecrep(yh_(:,i),nsp);
        else
            obs_pivotA = Y_(oA_IdxVec,1);      % extract pivot angle
            Y_(oA_IdxVec,1) = 0;
            Y_(oA_IdxVec,2:end) = subangle(Y_(oA_IdxVec,2:end),cvecrep(obs_pivotA,nsp-1));  % subtract pivot angle mod 2pi
            yh_(:,i) = W(1)*Y_(:,1) + W(2)*sum(Y_(:,2:nsp),2);
            yh_(oA_IdxVec,i) = 0;
            for k=2:nsp,
                yh_(oA_IdxVec,i) = addangle(yh_(oA_IdxVec,i), W(2)*Y_(oA_IdxVec,k));   % calculate UT mean ... mod 2pi
            end
            oFoo = cvecrep(yh_(:,i),nsp);
            temp2 = Y_ - oFoo;
            temp2(oA_IdxVec,:) = subangle(Y_(oA_IdxVec,:), oFoo(oA_IdxVec,:));
            yh_(oA_IdxVec,i) = addangle(yh_(oA_IdxVec,i), obs_pivotA);  % add pivot angle back to calculate actual predicted mean
        end


        [foo,Sy] = qr((sqrtW(2)*temp2(:,2:nsp))',0);       % QR update of observation error Cholesky factor. NOTE: here Sy
                                                           % is the UPPER Cholesky factor (Matlab excentricity)

        if possitive_W3                                    % deal with possible negative zero'th covariance weight
            Sy = cholupdate(Sy,sqrtW(3)*temp2(:,1),'+');
        else
            Sy = cholupdate(Sy,sqrtW(3)*temp2(:,1),'-');   % NOTE: here Sy  is the UPPER Cholesky factor (Matlab excentricity)
        end

        Sy = Sy';                                          % We need the lower triangular Cholesky factor

        Pxy = W(3)*temp1(:,1)*temp2(:,1)' + W(2)*temp1(:,2:nsp)*temp2(:,2:nsp)';

        KG = (Pxy/Sy')/Sy;


        if isempty(InferenceDS.innovation)
            inov(:,i) = obs(:,i) - yh_(:,i);
        else
            inov(:,i) = InferenceDS.innovation( InferenceDS, obs(:,i), yh_(:,i));  % inovation (observation error)
        end


        if isempty(sA_IdxVec)
           xh(:,i) = xh_(:,i) + KG*inov(:,i);
        else
           upd = KG*inov(:,i);
           xh(:,i) = xh_(:,i) + upd;
           xh(sA_IdxVec,i) = addangle(xh_(sA_IdxVec,i), upd(sA_IdxVec));
        end



        Sx_ = Sx_';

        cov_update_vectors = KG*Sy;      % Correct covariance. This is equivalent to :  Px = Px_ - KG*Py*KG';
        for j=1:Odim
            Sx_ = cholupdate(Sx_,cov_update_vectors(:,j),'-');
        end

        Sx = Sx_';
        state = xh(:,i);

        if pNoise.adaptMethod
        %--- update process noise if needed -----------------------
            switch pNoise.adaptMethod

            case 'anneal'
                dV = max(pNoise.adaptParams(1)*(dv.^2) , pNoise.adaptParams(2));
                ds = diag(Sx);
                dv = -ds + sqrt(dV + ds.^2);
                Sv = diag(dv);

            case 'robbins-monro'
                nu = 1/pNoise.adaptParams(1);
                dV = (1-nu)*(dv.^2) + nu*diag(KG*(KG*inov*inov')');
                ds = diag(Sx);
                dv = -ds + sqrt(dV + ds.^2);
                Sv = diag(dv);
                pNoise.adaptParams(1) = min(pNoise.adaptParams(1)+1, pNoise.adaptParams(2));

            otherwise
                error(' [ srukf ]  Process noise update method not allowed.');

            end

            pNoise.cov = Sv;
        %-----------------------------------------------------------
        end

    end   %... loop over all input vectors



otherwise
%===================================== STATE & JOINT ESTIMATION VERSION ===================================================

    L = Xdim + Vdim + Ndim;                                   % augmented state dimension
    nsp = 2*L+1;                                              % number of sigma-points
    kappa = alpha^2*(L+kappa)-L;                              % compound scaling parameter

    W = [kappa 0.5 0]/(L+kappa);                              % sigma-point weights
    W(3) = W(1) + (1-alpha^2) + beta;

    sqrtW = W;
    possitive_W3 = (W(3) > 0);                                % is zero'th covariance weight possitive?
    sqrtW(1:2) = sqrt(W(1:2));                                % square root weights
    sqrtW(3) = sqrt(abs(W(3)));

    Sqrt_L_plus_kappa = sqrt(L+kappa);

    Zeros_Xdim_X_Vdim     = zeros(Xdim,Vdim);
    Zeros_Vdim_X_Xdim     = zeros(Vdim,Xdim);
    Zeros_XdimVdim_X_Ndim = zeros(Xdim+Vdim,Ndim);
    Zeros_Ndim_X_XdimVdim = zeros(Ndim,Xdim+Vdim);

    Sx = Sstate;
    Sv = pNoise.cov;         % get process noise covariance Cholesky factor
    Sn = oNoise.cov;         % get observation noise covariance Cholesky factor
    mu_v = pNoise.mu;      % get process noise mean
    mu_n = oNoise.mu;      % get measurement noise mean


    if (U1dim==0), UU1=zeros(0,nsp); end
    if (U2dim==0), UU2=zeros(0,nsp); end


    % if process noise adaptation for joint estimation
    if pNoise.adaptMethod
        switch InferenceDS.inftype
          case 'joint'
            idx = pNoise.idxArr(end,:);     % get indeces of parameter block of combo-gaussian noise source
            ind1 = idx(1);                  % beginning index of parameter section
            ind2 = idx(2);                  % ending index of parameter section
            paramdim = ind2-ind1+1;         % infer parameter vector length
            dv = diag(Sv);                  % grab diagonal
            dv = dv(ind1:ind2);             % extract the part of the diagonal that relates to the 'parameter section'
          case 'state'
            ind1 = 1;
            ind2 = Xdim;
            paramdim = Xdim;
            dv = diag(Sv);
        end
    end


    %--- Loop over all input vectors -----------------------------------
    for i=1:NOV,

        if U1dim, UU1 = cvecrep(U1(:,i),nsp); end
        if U2dim, UU2 = cvecrep(U2(:,i),nsp); end

        %-----------------------------------------
        % TIME UPDATE

        Z   = cvecrep([state; mu_v; mu_n],nsp);
        Zm  = Z;                                         % copy needed for possible angle components section
        SzT = [Sx Zeros_Xdim_X_Vdim; Zeros_Vdim_X_Xdim Sv];
        Sz  = [SzT Zeros_XdimVdim_X_Ndim; Zeros_Ndim_X_XdimVdim Sn];
        sSz = Sqrt_L_plus_kappa * Sz;
        sSzM = [sSz -sSz];
        Z(:,2:nsp) = Z(:,2:nsp) + sSzM;

        %-- Calculate predicted state mean, dealing with angular discontinuities if needed
        if isempty(sA_IdxVec)
            X_ = InferenceDS.ffun( InferenceDS, Z(1:Xdim,:), Z(Xdim+1:Xdim+Vdim,:), UU1);  % propagate sigma-points through process model
            X_bps = X_;
            xh_(:,i) = W(1)*X_(:,1) + W(2)*sum(X_(:,2:nsp),2);
            temp1 = X_ - cvecrep(xh_(:,i),nsp);
        else
            Z(sA_IdxVec,2:nsp) = addangle(Zm(sA_IdxVec,2:nsp), sSzM(sA_IdxVec,:));      % fix sigma-point set for angular components
            X_ = InferenceDS.ffun( InferenceDS, Z(1:Xdim,:), Z(Xdim+1:Xdim+Vdim,:), UU1); % propagate sigma-points through process model
            X_bps = X_;
            state_pivotA = X_(sA_IdxVec,1);                                % extract pivot angle
            X_(sA_IdxVec,1) = 0;
            X_(sA_IdxVec,2:end) = subangle(X_(sA_IdxVec,2:end),cvecrep(state_pivotA,nsp-1));  % subtract pivot angle mod 2pi
            xh_(:,i) = W(1)*X_(:,1) + W(2)*sum(X_(:,2:nsp),2);
            xh_(sA_IdxVec,i) = 0;
            for k=2:nsp,
                xh_(sA_IdxVec,i) = addangle(xh_(sA_IdxVec,i), W(2)*X_(sA_IdxVec,k));     % calculate UT mean ... mod 2pi
            end
            sFoo = cvecrep(xh_(:,i),nsp);
            temp1 = X_ - sFoo;
            temp1(sA_IdxVec,:) = subangle(X_(sA_IdxVec,:), sFoo(sA_IdxVec,:));
            xh_(sA_IdxVec,i) = addangle(xh_(sA_IdxVec,i), state_pivotA);  % add pivot angle back to calculate actual predicted mean
        end

        [foo,Sx_] = qr((sqrtW(2)*temp1(:,2:nsp))',0);      % QR update of state Cholesky factor. NOTE: here Sx_
                                                           % is the UPPER Cholesky factor (Matlab excentricity)

        if possitive_W3                                    % deal with possible negative zero'th covariance weight
            Sx_ = cholupdate(Sx_,sqrtW(3)*temp1(:,1),'+');
        else
            Sx_ = cholupdate(Sx_,sqrtW(3)*temp1(:,1),'-'); % NOTE: here Sx_  is the UPPER Cholesky factor (Matlab excentricity)
        end


        Y_ = InferenceDS.hfun( InferenceDS, X_bps, Z(Xdim+Vdim+1:Xdim+Vdim+Ndim,:), UU2);


        %-- Calculate predicted observation mean, dealing with angular discontinuities if needed
        if isempty(oA_IdxVec)
            yh_(:,i) = W(1)*Y_(:,1) + W(2)*sum(Y_(:,2:nsp),2);
            temp2 = Y_ - cvecrep(yh_(:,i),nsp);
        else
            obs_pivotA = Y_(oA_IdxVec,1);      % extract pivot angle
            Y_(oA_IdxVec,1) = 0;
            Y_(oA_IdxVec,2:end) = subangle(Y_(oA_IdxVec,2:end),cvecrep(obs_pivotA,nsp-1));  % subtract pivot angle mod 2pi
            yh_(:,i) = W(1)*Y_(:,1) + W(2)*sum(Y_(:,2:nsp),2);
            yh_(oA_IdxVec,i) = 0;
            for k=2:nsp,
                yh_(oA_IdxVec,i) = addangle(yh_(oA_IdxVec,i), W(2)*Y_(oA_IdxVec,k));   % calculate UT mean ... mod 2pi
            end
            oFoo = cvecrep(yh_(:,i),nsp);
            temp2 = Y_ - oFoo;
            temp2(oA_IdxVec,:) = subangle(Y_(oA_IdxVec,:), oFoo(oA_IdxVec,:));
            yh_(oA_IdxVec,i) = addangle(yh_(oA_IdxVec,i), obs_pivotA);  % add pivot angle back to calculate actual predicted mean
        end

        [foo,Sy] = qr((sqrtW(2)*temp2(:,2:nsp))',0);       % QR update of observation error Cholesky factor. NOTE: here Sy
                                                           % is the UPPER Cholesky factor (Matlab excentricity)

        if possitive_W3                                    % deal with possible negative zero'th covariance weight
            Sy = cholupdate(Sy,sqrtW(3)*temp2(:,1),'+');
        else
            Sy = cholupdate(Sy,sqrtW(3)*temp2(:,1),'-');   % NOTE: here Sy  is the UPPER Cholesky factor (Matlab excentricity)
        end

        Sy = Sy';                                          % We need the lower triangular Cholesky factor


        %------------------------------------------------------
        % MEASUREMENT UPDATE

        Pxy = W(3)*temp1(:,1)*temp2(:,1)' + W(2)*temp1(:,2:nsp)*temp2(:,2:nsp)';

        KG = (Pxy/Sy')/Sy;

        if isempty(InferenceDS.innovation)
            inov(:,i) = obs(:,i) - yh_(:,i);
            if ~isempty(oA_IdxVec)
              inov(oA_IdxVec,i) = subangle(obs(oA_IdxVec,i), yh_(oA_IdxVec,i));
            end
        else
            inov(:,i) = InferenceDS.innovation( InferenceDS, obs(:,i), yh_(:,i));  % inovation (observation error)
        end


        if isempty(sA_IdxVec)
           xh(:,i) = xh_(:,i) + KG*inov(:,i);
        else
           upd = KG*inov(:,i);
           xh(:,i) = xh_(:,i) + upd;
           xh(sA_IdxVec,i) = addangle(xh_(sA_IdxVec,i), upd(sA_IdxVec));
        end

        cov_update_vectors = KG*Sy;      % Correct covariance. This is equivalent to :  Px = Px_ - KG*Py*KG';
        for j=1:Odim
            Sx_ = cholupdate(Sx_,cov_update_vectors(:,j),'-');
        end
        Sx = Sx_';

        state = xh(:,i);


        if pNoise.adaptMethod
        %--- update process noise if needed for joint estimation ----------------------
            switch pNoise.adaptMethod

            case 'anneal'
                dv = sqrt(max(pNoise.adaptParams(1)*(dv.^2) , pNoise.adaptParams(2)));
                Sv(ind1:ind2,ind1:ind2) = diag(dv);

            case 'robbins-monro'
                nu = 1/pNoise.adaptParams(1);
                subKG = KG(end-paramdim+1:end,:);
                dv = sqrt((1-nu)*(dv.^2) + nu*diag(subKG*(subKG*inov*inov')'));
                Sv(ind1:ind2,ind1:ind2) = diag(dv);
                pNoise.adaptParams(1) = min(pNoise.adaptParams(1)+1, pNoise.adaptParams(2));

            otherwise
                error(' [ srukf ]  Process noise update method not allowed.');

            end

            pNoise.cov = Sv;
        %-----------------------------------------------------------
        end

    end   %... loop over all input vectors


%====================================================================================================================
end


if (nargout>4),
    InternalVariablesDS.xh_   = xh_;
    InternalVariablesDS.Sx_   = Sx_;
    InternalVariablesDS.yh_   = yh_;
    InternalVariablesDS.inov  = inov;
    InternalVariablesDS.Sinov = Sy;
    InternalVariablesDS.KG    = KG;
end
