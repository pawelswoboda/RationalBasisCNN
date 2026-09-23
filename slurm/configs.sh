# Named backbone configurations: config_args <name> prints the flags for the
# experiment scripts. Every configuration listed here has logs in
# results/logs/ and appears in results/analyze.py.
config_args() {
  case "$1" in
    # ---- PascalVOC (experiments/pascal_voc.py, 2-D pseudo-coordinates) ----
    # ---- also SPair-71k (experiments/spair71k.py): same DGMC, same 2-D graphs
    spline)    echo "--backbone spline" ;;                       # SplineCNN k=5, K=25
    spline_k3) echo "--backbone spline --kernel_size 3" ;;       # K=9
    spline_k2) echo "--backbone spline --kernel_size 2" ;;       # K=4
    rational)    echo "--backbone rational" ;;                   # product basis, k=5, hat init
    rational_k3) echo "--backbone rational --kernel_size 3" ;;
    rational_mv)     echo "--backbone rational --rational_basis multivariate --degrees 8 6" ;;
    rational_mv_k3)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --kernel_size 3" ;;
    rational_mv_d54) echo "--backbone rational --rational_basis multivariate --degrees 5 4" ;;
    rational_mv_k3_gauss)   echo "--backbone rational --rational_basis multivariate --degrees 8 6 --kernel_size 3 --init gauss" ;;
    rational_mv_k3_cheb_vp) echo "--backbone rational --rational_basis multivariate --degrees 8 6 --kernel_size 3 --init cheb --vp" ;;
    mv_K1)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 1" ;;
    mv_K2)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 2" ;;
    mv_K4)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 4" ;;
    mv_K6)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 6" ;;
    mv_K9)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 9" ;;
    mv_K16) echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 16" ;;
    mv_K4_wd) echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 4 --basis_wd 1e-4" ;;
    # ---- regional basis: rationals confined to radial zones, K = zones * bases_per_zone ----
    zone_K4) echo "--backbone rational --rational_basis regional --zones 4 --bases_per_zone 1 --degrees 8 6 --vp" ;;
    zone_K9) echo "--backbone rational --rational_basis regional --zones 3 --bases_per_zone 3 --degrees 8 6 --vp" ;;
    # ---- fixed-basis controls: the shapes stay at init, only Theta trains ----
    mv_K4_frozen) echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 4 --freeze_basis" ;;
    mv_K9_frozen) echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 9 --freeze_basis" ;;
    mlp_K4) echo "--backbone rational --rational_basis mlp --vp --num_bases 4" ;;
    mlp_K9) echo "--backbone rational --rational_basis mlp --vp --num_bases 9" ;;
    # ---- FAUST (experiments/faust.py, 3-D pseudo-coordinates) ----
    faust_spline)             echo "--backbone spline" ;;        # k=5, K=125, add aggr, clip 1.0
    faust_spline_mean)        echo "--backbone spline --aggr mean" ;;
    faust_spline_mean_noclip) echo "--backbone spline --aggr mean --clip 0" ;;
    faust_spline_noclip)      echo "--backbone spline --clip 0" ;;  # literal PyG example
    faust_spline_pyginit)     echo "--backbone spline --pyg_init" ;;
    faust_spline_k3)          echo "--backbone spline --kernel_size 3" ;;
    faust_spline_k2)          echo "--backbone spline --kernel_size 2" ;;
    faust_rational_k3) echo "--backbone rational --kernel_size 3" ;;
    faust_mv_K4)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 4" ;;
    faust_mv_K8)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 8" ;;
    faust_mv_K16) echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 16" ;;
    faust_mv_K27) echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 27" ;;
    faust_mv_K4_mean) echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 4 --aggr mean" ;;
    faust_mlp_K8) echo "--backbone rational --rational_basis mlp --vp --num_bases 8" ;;
    # ---- N-Caltech101 / AEGNN (experiments/ncaltech101.py, 3-D pseudo-coordinates, k=2) ----
    ncal_pyg_spline)  echo "--backbone pyg_spline" ;;   # AEGNN as released (torch_spline_conv)
    ncal_spline)      echo "--backbone spline" ;;       # same operator, our implementation (K=8)
    ncal_spline_k3)   echo "--backbone spline --kernel_size 3" ;;  # K=27
    ncal_mv_K4)   echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 4" ;;
    ncal_mv_K8)   echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 8" ;;
    ncal_mv_K8_triton) echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 8 --basis_impl triton" ;;  # kernel validation run
    ncal_mv_K16)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 16" ;;
    ncal_mv_K8_d54) echo "--backbone rational --rational_basis multivariate --degrees 5 4 --init pca --vp --num_bases 8" ;;
    ncal_mv_K4_d54) echo "--backbone rational --rational_basis multivariate --degrees 5 4 --init pca --vp --num_bases 4" ;;
    ncal_rational_k2) echo "--backbone rational --kernel_size 2" ;;  # product basis, spline init, K=8
    ncal_mlp_K8)  echo "--backbone rational --rational_basis mlp --vp --num_bases 8" ;;
    ncal_pointnet) echo "--backbone pointnet" ;;        # Jeziorek et al. 2023 replacement
    ncal_spline_max) echo "--backbone spline --aggr max" ;;   # max instead of mean aggregation
    ncal_mv_K8_max)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 8 --aggr max" ;;
    ncal_pointnet_mean) echo "--backbone pointnet --pointnet_aggr mean" ;;
    ncal_spline_max_aug) echo "--backbone spline --aggr max --augment" ;;
    ncal_mv_K8_max_aug)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 8 --aggr max --augment" ;;
    # trained to convergence (submit with EPOCHS=200 as the cap): lr/10 on validation plateaus, stop after the 2nd decay plateaus
    ncal_spline_aug_conv)     echo "--backbone spline --augment --schedule plateau" ;;
    ncal_mv_K8_aug_conv)      echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 8 --augment --schedule plateau" ;;
    ncal_pointnet_aug_conv)   echo "--backbone pointnet --augment --schedule plateau" ;;
    ncal_spline_max_aug_conv) echo "--backbone spline --aggr max --augment --schedule plateau" ;;
    ncal_mv_K8_max_aug_conv)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 8 --aggr max --augment --schedule plateau" ;;
    ncal_spline_aug) echo "--backbone spline --augment" ;;   # + flip / translation augmentation
    ncal_mv_K8_aug)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 8 --augment" ;;
    ncal_pointnet_aug) echo "--backbone pointnet --augment" ;;
    # ---- N-Cars / AEGNN (experiments/ncaltech101.py --dataset ncars: r=3, 10k events, batch 64, 120x100) ----
    ncars_spline)     echo "--backbone spline" ;;
    ncars_mv_K4)      echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 4" ;;
    ncars_mv_K8)      echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 8" ;;
    ncars_pointnet)   echo "--backbone pointnet" ;;
    ncars_spline_max) echo "--backbone spline --aggr max" ;;
    ncars_mv_K8_max)  echo "--backbone rational --rational_basis multivariate --degrees 8 6 --init pca --vp --num_bases 8 --aggr max" ;;
    *) echo "unknown config '$1'" >&2; return 1 ;;
  esac
}
