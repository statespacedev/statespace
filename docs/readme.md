
<a name="190223"/>190223

kl-divergence for pf evaluation demonstrated below by three pf's in action during the first second of the jazwinksi problem - start-up and convergence. fundamental-units are hundreds of samples per dist at 100 hz, represented as dist-curves using kernel-density-estimation - green dist-curves for truth, blue dist-curves for pf. state-estimates are two red curves on the x,t-plane beneath the dist-curves.

![pf1](images/pf1.png)

![pf2](images/pf2.png)

![pf3](images/pf3.png)

190105

ukf adaptive jazwinksi switched to square-root filtering, qr-factorizaion, cholesky-factor update and downdate. improved numerical stability and scaled sampling is clear. still a question around scalar-obs and the obs cholesky-factor and gain. with an adhoc stabilizer on the obs cholesky-factor it's working well overall.

181230

pf adaptive jazwinksi. parameter-roughening.

181226

ukf adaptive jazwinski. sample-and-propagate tuning.

180910

ekf adaptive jazwinski. ud-factorized square-root filtering required for numerical stability.
