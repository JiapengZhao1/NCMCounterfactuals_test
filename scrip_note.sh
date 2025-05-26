estimation for 8exp
python -m src.main est_exp1_1 gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 1 --gpu 0

10 trails
python -m src.main est_8exp_10t gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 8exp -t 10 -n 100 -d 1 --gpu 0
python -m src.main est_8exp_10t gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 8exp -t 10 -n 1000 -d 1 --gpu 0

large 10 trails
python -m src.main est_large_10t gan --lr 0.00002 --data-bs 1000 --ncm-bs 1000 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G large -t 10 -n 1000 -d 1 --gpu 0
python -m src.main est_large_10t gan --lr 0.00002 --data-bs 4096 --ncm-bs 4096 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G large -t 10 -n 10000 -d 1 --gpu 0



00how c
# estimation for CH 
python -m src.main est_5ch_2 gan --lr 0.00002 --data-bs 1000 --ncm-bs 1000 --h-size 64 --u-size 1 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G 5ch -t 1 -n 1000 -d 1 --gpu 0
python -m src.main est_5ch_2 gan --lr 0.00002 --data-bs 4096 --ncm-bs 4096 --h-size 64 --u-size 1 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G 5ch -t 1 -n 10000 -d 1 --gpu 0

#name    Number of parameters
#5-ch      129k
#9-ch      383k
#25-ch     2.7M
#49-ch     10.2M
#99-ch     41.3M

# estimation for dimond 
python -m src.main est_3d_2 gan --lr 0.00002 --data-bs 1000 --ncm-bs 1000 --h-size 64 --u-size 1 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G 3d -t 1 -n 1000 -d 1 --gpu 0
python -m src.main est_3d_2 gan --lr 0.00002 --data-bs 4096 --ncm-bs 4096 --h-size 64 --u-size 1 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G 3d -t 1 -n 10000 -d 1 --gpu 0

#9-d      383k
#17-d     1.3M
#65-d     17.9M

# estimation for corn cloud 
python -m src.main est_2cc_2 gan --lr 0.00002 --data-bs 1000 --ncm-bs 1000 --h-size 64 --u-size 1 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G 2cc -t 1 -n 1000 -d 1 --gpu 0
python -m src.main est_2cc_2 gan --lr 0.00002 --data-bs 4096 --ncm-bs 4096 --h-size 64 --u-size 1 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G 2cc -t 1 -n 10000 -d 1 --gpu 0
 


#6-cc      181k     
#15-cc     1M














