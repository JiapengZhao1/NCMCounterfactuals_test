#estimation for 8exp
python -m src.main est_exp1_2 gan --lr 0.00002 --gen CPT --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 1 --gpu 0

#10 trails for 8exp
python -m src.main est_8exp_10t gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 8exp -t 10 -n 100 -d 1 --gpu 0
python -m src.main est_8exp_10t gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 8exp -t 10 -n 1000 -d 1 --gpu 0

#6large 10 trails
python -m src.main est_6large_5t gan --lr 0.00002 --data-bs 1000 --ncm-bs 1000 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 6large -t 10 -n 1000 -d 1 --gpu 0
python -m src.main est_6large_5t gan --lr 0.00002 --data-bs 4096 --ncm-bs 4096 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G large -t 10 -n 10000 -d 1 --gpu 0



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


#estimation for exp1 with dim = 2
python -m src.main est_exp1_dim2 gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 2 --gpu 0


#check if it can reload checkpoints
python -m src.main est_exp1_2 gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 1 --gpu 0



#updated MAD calculation

#10 trails for 8exp
python -m src.main est_8exp_10t_1 gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 8exp -t 10 -n 100 -d 1 --gpu 0
python -m src.main est_8exp_10t_1 gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 8exp -t 10 -n 1000 -d 1 --gpu 0

#6large 10 trails
python -m src.main est_6large_5t_1 gan --lr 0.00002 --data-bs 1000 --ncm-bs 1000 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 6large -t 5 -n 1000 -d 1 --gpu 0
python -m src.main est_6large_5t_1 gan --lr 0.00002 --data-bs 4096 --ncm-bs 4096 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 6large -t 5 -n 10000 -d 1 --gpu 0


#estimation for exp1 with dim = n
python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 1 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 1 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 2 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 3 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 3 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 4 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 4 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 5 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 5 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 6 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 6 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 7 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 7 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 8 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 8 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 9 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 9 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 10 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 10 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 11 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 11 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 12 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 12 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 13 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 13 --gpu 0

python -m src.main est_exp1_n_dim_ gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 14 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 14 --gpu 0



#MLE exp1 try
python -m src.main mle_exp1_test1 mle --full-batch --gen CPT --h-size 64 --query-track avg_error --max-query-iters 1000 --mc-sample-size 1000 -G exp1 -t 1 -n 100 -d 1 --gpu 0
python -m src.main mle_exp2_test1 mle --full-batch --gen CPT --h-size 64 --query-track avg_error --max-query-iters 1000 --mc-sample-size 1000 -G exp2 -t 1 -n 100 -d 1 --gpu 0

#MLE exp8 10 trails
python -m src.main mle_8exp_cpt1 mle --full-batch --gen CPT --h-size 64 --query-track avg_error --max-query-iters 1000 --mc-sample-size 10000 -G 8exp -t 10 -n 100 -d 1 --gpu 0
python -m src.main mle_8exp_cpt1 mle --full-batch --gen CPT --h-size 64 --query-track avg_error --max-query-iters 1000 --mc-sample-size 10000 -G 8exp -t 10 -n 1000 -d 1 --gpu 0

#GAN 8exp 10 trails
python -m src.main est_8exp_cpt1 gan --gen CPT --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 8exp -t 10 -n 100 -d 1 --gpu 0
python -m src.main est_8exp_cpt1 gan --gen CPT --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 8exp -t 10 -n 1000 -d 1 --gpu 0

#MLE 6large 5 trails
#test
python -m src.main mle_6large_cpt1 mle --data-bs 128 --ncm-bs 128 --gen CPT --h-size 64 --query-track avg_error --max-query-iters 1000 --mc-sample-size 200 -G 6large -t 1 -n 1000 -d 1 --gpu 0


python -m src.main mle_6large_cpt1 mle --data-bs 128 --ncm-bs 128 --gen CPT --h-size 64 --query-track avg_error --max-query-iters 1000 --mc-sample-size 1000 -G 6large -t 5 -n 1000 -d 1 --gpu 0
python -m src.main mle_6large_cpt1 mle --data-bs 4096 --ncm-bs 4096 --gen CPT --h-size 64 --query-track avg_error --max-query-iters 1000 --mc-sample-size 1000 -G 6large -t 5 -n 10000 -d 1 --gpu 0

#GAN 6large 5 trails
python -m src.main est_6large_cpt1 gan --gen CPT --lr 0.00002 --data-bs 1000 --ncm-bs 1000 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 6large -t 5 -n 1000 -d 1 --gpu 0
python -m src.main est_6large_cpt1 gan --gen CPT --lr 0.00002 --data-bs 4096 --ncm-bs 4096 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track avg_error --single-disc --gen-sigmoid --mc-sample-size 10000 -G 6large -t 5 -n 10000 -d 1 --gpu 0
