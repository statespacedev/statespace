
190223

![pf1](images/pf1.png)

implement kl-divergence distance-measure for pf performance eval.

190105

ukf adaptive jazwinksi switched to square-root filtering, qr-factorizaion, cholesky-factor update and downdate. improved numerical stability and scaled sampling is pretty clearly working correctly. still a question around scalar-obs and the obs cholesky-factor and gain. with an adhoc stabilizer on the obs cholesky-factor, it's working well overall.

181230

pf adaptive jazwinksi. parameter-roughening.

181226

ukf adaptive jazwinski. sample-and-propagate tuning.

180910

ekf adaptive jazwinski. ud-factorized square-root filtering required for numerical stability.
